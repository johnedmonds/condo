import polars as pl
import calendar

# Source: BLS data set CUUR0000SA0 https://data.bls.gov/timeseries/CUUR0000SA0
# Allows us to update historical months to match the current month (e.g. when sampling from them to predict next year's budget).
def load_cpiu() -> pl.DataFrame:
    return (
        pl.read_excel("cpiu.xlsx", read_options={"header_row": 11})
        .rename({m: str(i) for i, m in enumerate(calendar.month_abbr[1:])})
        .unpivot(
            map(str, range(12)),
            index=["Year", "Annual"],
            variable_name="month",
            value_name="cpiu",
        )
        .sort("Year")
        .with_columns(
            pl.col("month").str.to_integer(),
            annual_computed=pl.col("cpiu").mean().over("Year"),
        )
        .with_columns(
            annual_delta=pl.when(
                pl.col("annual_computed").shift(-1) != pl.col("annual_computed")
            )
            .then(
                (pl.col("annual_computed").shift(-1) - pl.col("annual_computed"))
                / pl.col("annual_computed")
            )
            .backward_fill()
            .forward_fill()
        )
        .with_columns(
            filled_cpiu=pl.coalesce(
                pl.col("cpiu"),
                (1 + pl.col("annual_delta") / 12).pow(pl.col("month"))
                * pl.col("cpiu").first().over("Year"),
            )
        )
        .with_columns(
            # We want month to be 0-based when doing the exponentiation above but need it to be 1-based for joins
            month=pl.col("month") + 1,
            eoy_delta=(pl.col("filled_cpiu").last() - pl.col("filled_cpiu"))
            / pl.col("filled_cpiu"),
        )
    )
