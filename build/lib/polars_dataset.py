import logging
from pathlib import Path

import lmfit as lm
import numpy as np
import polars as pl
import polars_splines
from scipy.interpolate import PchipInterpolator


class Dataset:
    def __init__(self, data, index: str, id_vars: list[str] = None):
        self._id_vars = id_vars or []
        self._index = index
        if isinstance(data, type(self)):
            self._id_vars = id_vars or data._id_vars
            self._df = data._df
        else:
            self._df = self._initialize_df(data)
            if isinstance(data, list) and all(isinstance(d, type(self)) for d in data):
                if len({d.index for d in data}) > 1:
                    raise ValueError("All datasets must have the same index")
                self._id_vars = self._combine_id_vars(data)
        self._verify()

    def _initialize_df(self, data):
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, pl.LazyFrame):
            return data.collect()
        elif isinstance(data, (list, tuple, set)):
            combined_data = [
                self._initialize_df(d)
                for d in data
                if isinstance(d, (pl.DataFrame, type(self)))
            ]
            return pl.concat(combined_data) if combined_data else None
        elif isinstance(data, type(self)):
            return data.df
        else:
            raise TypeError(
                "Data must be of type Dataset, DataFrame, LazyFrame, or a uniform iterable of them"
            )

    def _verify(self):
        if self.index not in self.df.columns:
            raise ValueError(f"`{self._index}` not in DataFrame.")
        if any(id not in self.df.columns for id in self.id_vars):
            raise ValueError("All `id_vars` must exist in DataFrame.")

    def __getattr__(self, attr):
        attribute = getattr(self.df, attr)
        if callable(attribute):
            return self._wrap_method(attr)
        return attribute

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

    def __getitem__(self, item):
        return self.df[item]

    def __str__(self):
        return str(self.df)

    def __dataframe__(self, **kwargs):
        return self.df.__dataframe__(**kwargs)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        if not isinstance(value, pl.DataFrame):
            raise TypeError("`df` must be of type polars.DataFrame")
        if self._index and self._index not in value.columns:
            raise ValueError(f"The transformation does not preserve `{self._index}`")
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
            raise ValueError("All `id_vars` must be in DataFrame columns")
        self._id_vars = value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if value not in self.df.columns:
            raise ValueError(f"`{value}` not in DataFrame")
        self._index = value

    def _combine_id_vars(self, ds_list):
        return list(set().union(*[d.id_vars for d in ds_list]))

    def set(self, index=None, id_vars=None):
        if index:
            self.index = index
        if id_vars:
            self.id_vars = id_vars
        return self

    def select(self, *args):
        ds = self.clone()
        ds.df = ds.df.select(*args)
        return ds

    def select_data(self, *args):
        ds = self.clone()
        cols = [ds.index] + ds.id_vars
        ds.df = ds.df.select(*cols, *args)
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
            ds.df = result
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
        ds.df = df
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
            args = (self.id_vars or []) + list(args) + [self.index]
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


# def interpolate_frame(df, x, id, value_vars, interpolator=PchipInterpolator):
#     xname, xvalues = x
#     id_vars, id_vals = id
#     if not id_vals:
#         id_vals = list(
#             map(
#                 lambda s: s.item(),
#                 df.select(pl.col(name).first().alias(name) for name in id_vars),
#             )
#         )

#     x = np.asarray(df[xname])
#     if len(x) == 1:
#         df_interp = pl.DataFrame({name: np.asarray(df[name]) for name in value_vars})
#     else:
#         df_interp = pl.DataFrame(
#             {
#                 name: interpolator(x, np.asarray(df[name]))(xvalues)
#                 for name in value_vars
#             }
#         )
#     return df_interp.with_columns(
#         pl.lit(val).alias(key)
#         for key, val in {id: val for id, val in zip(id_vars, id_vals)}.items()
#     ).with_columns(pl.Series(xname, xvalues))


# def autophase(X, Y, return_phase=False, return_Y=True):
#     """Automatically adjust the phase of a complex signal."""

#     def objective(params, X, Y):
#         phase = params["phase"]
#         Y_new = X * np.sin(phase) + Y * np.cos(phase)
#         return Y_new**2

#     params = lm.Parameters()
#     params.add("phase", value=0)

#     res = lm.minimize(
#         objective,
#         params=params,
#         args=(
#             X,
#             Y,
#         ),
#     )

#     phase = res.params["phase"].value
#     X_new = X * np.cos(phase) - Y * np.sin(phase)
#     Y_new = X * np.sin(phase) + Y * np.cos(phase)

#     returns = (X_new,)
#     if return_Y:
#         returns += (Y_new,)
#     if return_phase:
#         returns += (phase,)
#     return returns


# def zero_quadrature(x: pl.Series):
#     """
#     Zero the quadrature component of a lock-in signal.

#     Parameters
#     ----------
#     x : pl.Series[pl.Struct]
#         The lock-in signal, with two fields: the X and Y channel.

#     Returns
#     -------
#     pl.Series[pl.Float64]
#         The in-phase component of the lock-in signal.
#     """
#     return autophase(
#         x.struct[0],
#         x.struct[1],
#         return_phase=False,
#         return_Y=False,
#     )[0]


# def create_dataset(
#     paths: pl.DataFrame,
#     column_names: list[str],
#     index: str,
#     lockin_schema: dict[str, tuple[str, str] | str],
#     id_schema: dict[pl.DataType] = None,
#     **kwargs,
# ) -> Dataset:
#     """Create a Dataset from a list of data files.

#     Parameters
#     ----------
#     paths : pl.DataFrame
#         A DataFrame with one column containing the paths to the files.
#     column_names : list[str]
#         A list of column names to use for the data.
#     index : str
#     lockin_schema : dict[tuple[str] | str, str]
#         Data files may contain two channels (X and Y). If so, the relative phase
#         will be adjusted to maximize the amplitude of the X channel, and only X will be
#         retained. The dictionary entry of this data has the structure
#             - key (str): new name for retained column
#             - value (tuple[str]): 2-tuple of names of X and Y channel, respectively.
#         If only one single channel, the dictionary entry has the structure
#             - key (str): new name for column
#             - value (str): name of column in data file
#     id_schema : dict[pl.DataType], optional
#         Names and types of the columns of `path` representing id (not index) parameters.

#     Returns
#     -------
#     Dataset
#         A Dataset object.
#     """

#     kwargs = {
#         "separator": kwargs.pop("separator", "\t"),
#         "has_header": kwargs.pop("has_header", False),
#         "comment_prefix": kwargs.pop("comment_prefix", "#"),
#         **kwargs,
#     }

#     pair_dict = {k: v for k, v in lockin_schema.items() if isinstance(v, tuple)}
#     lone_dict = {k: v for k, v in lockin_schema.items() if isinstance(v, str)}

#     lockin_exprs = [
#         pl.struct(x, y).map_batches(zero_quadrature).alias(name)
#         for name, (x, y) in pair_dict.items()
#     ]
#     lockin_exprs += [pl.col(col).alias(name) for name, col in lone_dict.items()]
#     frames = []
#     id_schema = id_schema or {}
#     for *idvals, path in paths.iter_rows():
#         id_exprs = [
#             pl.lit(val).cast(dtype).alias(name)
#             for val, (name, dtype) in zip(idvals, id_schema.items())
#         ]

#         df = (
#             pl.read_csv(path, new_columns=column_names, **kwargs)
#             .with_columns(*id_exprs, index, *lockin_exprs)
#             .select(
#                 *(pl.col(name) for _, (name, _) in zip(idvals, id_schema.items())),
#                 pl.col(index),
#                 *(pl.col(name) for name in lockin_schema.keys()),
#             )
#             .sort(index, *id_schema.keys())
#         )
#         frames.append(df)
#     return Dataset(frames, index, list(id_schema))
