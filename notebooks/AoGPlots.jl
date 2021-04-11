### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ a7aef68a-9adc-11eb-24a5-eda95afc54e6
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 8b38927e-11f1-45fd-b630-44113f0aac3d
using CairoMakie   # for displaying static plots in a Pluto notebook

# ╔═╡ 35b834be-55e1-4a86-8dd8-f1f57260f3bd
using Arrow, AlgebraOfGraphics, DataFrames, Statistics

# ╔═╡ 45e5fcb6-be53-4c03-8415-173578219d06
md"""
# Creating multi-panel plots

This notebook shows creating a multi-panel plot similar to Figure 2 of Fühner, Golle, Granacher, & Kliegl (2021).

The data have been saved as an Arrow-format file.
"""

# ╔═╡ a8bc0d0e-fc13-4196-95c9-5922ed400cb4
dat = Arrow.Table("./data/fggk21.arrow");

# ╔═╡ 4825ea74-1902-4423-883d-a7b00a2aa359
typeof(dat)

# ╔═╡ 52b65f0d-f09e-47c8-81b5-624a09bc1c9a
md"""
## Creating a summary data frame

The response to be plotted is the mean score by `Test` and `Sex` and `age`, rounded to the nearest 0.1 years.

The first task is to round the `age` to 1 digit after the decimal place, which can be done with `select` applied to a `DataFrame`.
In some ways this is the most complicated expression in creating the plot so we will break it down.
`select` is applied to `DataFrame(dat)`, which is the conversion of the `Arrow.Table`, `dat`, to a `DataFrame`.
This is necessary because an `Arrow.Table` is immutable but a `DataFrame` can be modified.

The arguments after the `DataFrame` describe how to modify the contents.
The first `:` indicates that all the existing columns should be included.
The other expression can be pairs (created with the `=>` operator) of the form `:col => function` or of the form `:col => function => :newname`.
(See the [documentation of the DataFrames package](http://juliadata.github.io/DataFrames.jl/stable/) for details.)

In this case the function is an anonymous function of the form `round.(x, digits=1)` where "dot-broadcasting" is used to apply to the entire column (see [this documentation](https://docs.julialang.org/en/v1/manual/functions/#man-vectorized) for details).
"""

# ╔═╡ 53da799e-f794-42a5-9255-6503fde6dfa0
describe(select(DataFrame(dat), :, :age => (x -> round.(x, digits=1)) => :rnd_age))

# ╔═╡ fa5e6e67-d37b-414d-8f3a-34f623f423af
md"""
The next stage is a *group-apply-combine* operation to group the rows by `Sex`, `Test` and `rnd_age` then apply `mean` to the `zScore` and also apply `length` to `zScore` to record the number in each group.
"""

# ╔═╡ f27eab8e-5fbd-442c-b4a2-a95b4c29cd52
df = combine(
	groupby(
		select(DataFrame(dat), :, :age => (x -> round.(x, digits=1)) => :rnd_age),
		[:Sex, :Test, :rnd_age],
	),
	:zScore => mean,
	:zScore => length => :n,
)

# ╔═╡ e61f86b4-2e9e-4522-af2f-b97097e4c95c
md"""
## Creating the plot

The `AlgebraOfGraphics` package applies operators to the results of functions such as `data` (specify the data table to be used), `mapping` (designate the roles of columns), and `visual` (type of visual presentation).
"""

# ╔═╡ 999c343f-c0ee-4e91-9596-7f30ceca117c
data(df) * 
mapping(:rnd_age, :zScore_mean, layout_x=:Test, color = :Sex) *
visual(Scatter) |> draw()

# ╔═╡ Cell order:
# ╠═a7aef68a-9adc-11eb-24a5-eda95afc54e6
# ╟─45e5fcb6-be53-4c03-8415-173578219d06
# ╠═8b38927e-11f1-45fd-b630-44113f0aac3d
# ╠═35b834be-55e1-4a86-8dd8-f1f57260f3bd
# ╠═a8bc0d0e-fc13-4196-95c9-5922ed400cb4
# ╠═4825ea74-1902-4423-883d-a7b00a2aa359
# ╟─52b65f0d-f09e-47c8-81b5-624a09bc1c9a
# ╠═53da799e-f794-42a5-9255-6503fde6dfa0
# ╟─fa5e6e67-d37b-414d-8f3a-34f623f423af
# ╠═f27eab8e-5fbd-442c-b4a2-a95b4c29cd52
# ╟─e61f86b4-2e9e-4522-af2f-b97097e4c95c
# ╠═999c343f-c0ee-4e91-9596-7f30ceca117c
