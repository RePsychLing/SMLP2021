using DataFrames
using Chain
using DataFrameMacros

using CSV
using Arrow
using Downloads

using StatsBase
using Dates

using MacroTools: prettify


url = "https://github.com/RePsychLing/SMLP2021/raw/main/notebooks/data/fggk21.arrow"



df = DataFrame(Arrow.Table(Downloads.download(url)))

df = url |>
    Downloads.download |>
    Arrow.Table |>
    DataFrame |>
    x -> CSV.write("test.csv", x)

df = @chain url begin
    Downloads.download
    Arrow.Table
    DataFrame
    CSV.write("test.csv", _)
end


@macroexpand(@chain url begin
    Downloads.download
    Arrow.Table
    DataFrame
end) |> prettify





describe(df)


transform, select, groupby, combine, subset


transform(df, :Sex =>
    ByRow(bla -> bla == "female" ? "girl" : "boy") =>
    :type)




@transform(df, :type = :Sex == "female" ? "girl" : "boy")






@groupby(df, :age = round(Int, :age))

df.age
transform(df, :age => (col -> col .+ 1) => :ageplus)

@transform(df, :age + 1)
@transform(df, @c :age .- mean(:age))



@chain df begin
    @transform(:type = :Sex == "female" ? "girl" : "boy")
    @groupby(:half = :zScore > 0 ? "upper" : "lower", :type)
    combine(nrow => :n)
    # @aside CSV.write("test.csv", _)
    @transform(:nsquared = :n ^ 2)
end
