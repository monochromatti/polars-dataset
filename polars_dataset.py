import logging
from itertools import chain
from pathlib import Path

import polars as pl
import polars_splines
from bs4 import BeautifulSoup
from polars.exceptions import ColumnNotFoundError


class Dataset:
    def __init__(self, data, index: str, id_vars: list[str] = None):
        self._index = index
        self._id_vars = id_vars or []

        if isinstance(data, type(self)):
            self._id_vars = id_vars or data._id_vars
            self._df = data._df
        else:
            if isinstance(data, list | tuple) and all(
                isinstance(d, type(self)) for d in data
            ):
                if len({d.index for d in data}) > 1:
                    raise ValueError("All datasets must have the same index")
                self._id_vars = list(set().union(*[d.id_vars for d in data]))
            self._df = self._init_df(data)
            if isinstance(self._df, list):
                columns = self._id_vars + [self._index]
                try:
                    self._df = pl.concat(
                        (
                            df.select(*columns, pl.all().exclude(columns))
                            for df in self._df
                        )
                    )
                except ColumnNotFoundError as e:
                    cols = ", ".join(columns)
                    raise ColumnNotFoundError(
                        f"Missing from column(s): {e}. All datasets must contain: {cols}."
                    ) from e

    def _init_df(self, data):
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, pl.LazyFrame):
            return data.collect()
        elif isinstance(data, type(self)):
            return data.df
        elif isinstance(data, list | tuple):
            combined_data = [
                self._init_df(d)
                for d in data
                if isinstance(d, type(self) | pl.DataFrame | pl.LazyFrame)
            ]
            return combined_data
        else:
            raise TypeError(
                "Data must be of type (Dataset | DataFrame | LazyFrame) or a uniform iterable of them"
            )

    def _wrap_method(self, method_name):
        method = getattr(self.df, method_name)

        def wrapped_method(*args, **kwargs):
            result = method(*args, **kwargs)
            if isinstance(result, pl.DataFrame):
                ds = Dataset(result, index=self.index, id_vars=self.id_vars)
                ds.df = result
                return ds
            return result

        return wrapped_method

    def __getattr__(self, attr):
        attribute = getattr(self.df, attr)
        if callable(attribute):
            return self._wrap_method(attr)
        return attribute

    def __getitem__(self, item):
        return self.df[item]

    def __str__(self):
        return str(self.df)

    def _repr_html_(self):
        html_string = self.df._repr_html_()
        soup = BeautifulSoup(html_string, "html.parser")
        pos_index = self.df.columns.index(self.index)
        id_index = [self.df.columns.index(id) for id in self.id_vars]
        for row in soup.select("table.dataframe tbody tr"):
            cells = row.find_all("td")
            cells[pos_index]["style"] = "background-color: rgba(0,128,0,0.1);"
            for i in id_index:
                cells[i]["style"] = "background-color: rgba(0,128,255,0.1);"
        return str(soup)

    def __dataframe__(self, **kwargs):
        return self.df.__dataframe__(**kwargs)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        if not isinstance(value, pl.DataFrame):
            raise TypeError("Only a polars.DataFrame can be assigned.")
        if self._index and self._index not in value.columns:
            raise ValueError(
                f"The transformation does not preserve the index, `{self._index}`"
            )
        if self._id_vars:
            self._id_vars = [id for id in self._id_vars if id in value.columns]
        self._df = value

    @property
    def id_vars(self):
        return self._id_vars

    @id_vars.setter
    def id_vars(self, value):
        value = value or []
        if any(id not in self.df.columns for id in value):
            raise ValueError("The assignment involves unknown column names.")
        self._id_vars = value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if value not in self.df.columns:
            raise ValueError(f"The column `{value}` is not in DataFrame")
        self._index = value

    def set(self, index=None, id_vars=None):
        if index:
            self.index = index
        if id_vars:
            self.id_vars = id_vars
        return self.sort_columns()

    def select(self, *args):
        ds = self.clone()
        ds.df = ds.df.select(*args)
        return ds

    def pivot(self, *args, **kwargs):
        return self.df.clone().pivot(*args, **kwargs)

    def select_data(self, *args):
        ds = self.clone()
        retained_columns = ds.df.select(pl.col(ds.id_vars + [ds.index]))
        new_columns = ds.df.select(*args)
        ds.df = retained_columns.hstack(new_columns)
        return ds

    def fetch(self, *args):
        return self.df.clone().select(*args)

    @property
    def value_vars(self):
        return [
            col
            for col in self.df.columns
            if col not in (set(self.id_vars) | {self.index})
        ]

    def join(self, ds, **kwargs):
        if isinstance(ds, type(self)):
            left_idvars = self.id_vars or []
            right_idvars = ds.id_vars or []
            id_vars = left_idvars + [
                var for var in right_idvars if var not in left_idvars
            ]
            return Dataset(
                self.df.lazy().join(ds.df.lazy(), **kwargs).collect(),
                index=self.index,
                id_vars=id_vars,
            )
        return Dataset(
            self.df.lazy().join(ds.lazy(), **kwargs).collect(),
            index=self.index,
            id_vars=self.id_vars,
        )

    def rename(self, mapping: dict[str, str]):
        return Dataset(
            self.clone().df.rename(mapping),
            mapping.get(self.index, self.index),
            id_vars=[mapping.get(id, id) for id in (self.id_vars or [])],
        )

    def pipe(self, func, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if not isinstance(result, type(self)):
            ds = self.clone()
            ds.df = result.select(ds.id_vars, ds.index, ds.value_vars)
            return ds
        return result

    def _apply_spline(self, group, xi, value_vars, id_vars, **kwargs):
        id_vals = group.select(pl.col(*id_vars).first())
        group = group.select(
            pl.struct(xi.name, col).splines.spline(xi=list(xi), **kwargs).alias(col)
            for col in value_vars
        ).with_columns(xi, *id_vals)
        return group

    def regrid(self, x, **kwargs):
        if not isinstance(x, pl.Series):
            raise TypeError("`x` must be a polars.Series")

        ds = self.clone()
        df, schema_map = self._unnest_structs(ds.df)

        id_vars = ds.id_vars.copy()
        if x.name in id_vars:
            id_vars.remove(x.name)
            id_vars.append(ds.index)
        value_vars = [var for var in df.columns if var not in id_vars and var != x.name]
        if ds.id_vars:
            df = df.group_by(*id_vars, maintain_order=True).map_groups(
                lambda group: self._apply_spline(
                    group, x, value_vars, id_vars, **kwargs
                )
            )
        else:
            df = df.select(
                pl.struct(x.name, col).splines.spline(xi=list(x), **kwargs).alias(col)
                for col in value_vars
            ).with_columns(x)

        df = self._rebuild_structs(df, schema_map)
        ds.df = df.select(ds.columns)
        return ds

    def _unnest_structs(self, df):
        schema_map = {}
        for name, dtype in df.schema.items():
            if isinstance(dtype, pl.Struct):
                init_schema = dtype.to_schema()
                temp_schema = {f"{name}.{k}": v for k, v in init_schema.items()}
                schema_map[name] = {
                    "fields": list(init_schema.keys()),
                    "columns": list(temp_schema.keys()),
                }
                df = df.with_columns(
                    pl.col(name).struct.rename_fields(schema_map[name]["columns"])
                )
        return df.unnest(schema_map.keys()), schema_map

    def _rebuild_structs(self, df, schema_map):
        return df.with_columns(
            pl.struct(schema_map[name]["columns"])
            .struct.rename_fields(schema_map[name]["fields"])
            .alias(name)
            for name in schema_map.keys()
        ).drop(
            [col for name in schema_map.keys() for col in schema_map[name]["columns"]]
        )

    def drop(self, names):
        if isinstance(names, str):
            names = [names]
        if self.index in names:
            raise ValueError("Cannot drop the `index` column")
        ds = self.clone()
        ds.df = ds.df.drop(names)
        return ds

    def coord(self, name) -> pl.Series:
        return self.df[name].unique(maintain_order=True)

    def extrema(self, colname):
        return self.df.select(
            pl.col(colname).min().alias("min"), pl.col(colname).max().alias("max")
        ).row(0)

    def sort(self, *args, auto=True, **kwargs):
        if (not args) and auto:
            args = (self.id_vars or []) + [self.index] + list(args)
        ds = self.clone()
        ds.df = ds.df.sort(*args, **kwargs)
        return ds

    def sort_columns(self):
        ds = self.clone()
        ds.df = ds.df.select(*self.id_vars, self.index, *self.value_vars)
        return ds

    def drop_nan(self):
        ds = self.clone()
        df, schema_map = self._unnest_structs(ds.df)
        df = (
            df.with_columns(
                pl.any_horizontal(pl.all().is_nan()).alias("is_nan"),
            )
            .filter(~pl.col("is_nan"))
            .drop("is_nan")
        )
        df = self._rebuild_structs(df, schema_map)
        ds.df = df
        return ds


class Datafile:
    def __init__(self, path: str | Path, index: str = None, id_vars: list[str] = None):
        self.path = path if isinstance(path, Path) else Path(path)
        self.name = self.path.stem
        self.index = index
        self.id_vars = id_vars

    def load(self):
        try:
            if self.index is None:
                return pl.read_csv(self.path)
            else:
                return Dataset(pl.read_csv(self.path), self.index, self.id_vars)
        except FileNotFoundError:
            logging.error(f"Could not find {self.path}")

    def write(self, data: pl.DataFrame | Dataset):
        data.write_csv(self.path)

    def __str__(self):
        details = [
            f"filename: {self.name}",
            f"path: {self.path}",
            f"index: {self.index if self.index else 'None'}",
            f"id_vars: {', '.join(self.id_vars) if self.id_vars else 'None'}",
        ]
        return "\n".join(details)

    def __repr__(self):
        return str(self)
