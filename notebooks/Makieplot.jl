### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 0c8c6b94-a059-11eb-196f-cde6617ea966
begin
	using Pkg
	Pkg.activate(".")
	Pkg.add(["Arrow", "CairoMakie", "DataFrames"])
	Pkg.instantiate()
end

# ╔═╡ 379d2a3d-17ee-4173-b0f0-96ec639ec299
using Arrow, CairoMakie, DataFrames, Statistics

# ╔═╡ 7f109de7-2565-4ab7-97a6-58c52f583257
md"Ensure that the required packages are available"

# ╔═╡ e75bd3d6-c55b-4d30-903a-673cae83af2a
md"and load the packages to be used. (`Statistics` is in the standard library and always available.)"

# ╔═╡ 68ef0bef-606f-41e8-b258-69a554fe6935
md"Retrieve the data table from the Arrow-formatted file."

# ╔═╡ c35830ca-891e-4ac2-85e0-9c1d386e39dd
tbl = Arrow.Table("./data/fggk21.arrow")

# ╔═╡ 2fac75f6-e1cd-4727-b30d-c8e7952e8862
md"""
We want the levels of `Test` to be in a particular order as they were stored from R.
This ordering of levels is preserved in the Arrow table
"""

# ╔═╡ 9059f205-a459-422e-858d-0fbb103c8430
levels(tbl.Test)  # Arrow preserves order of levels from R factor

# ╔═╡ 2db1e041-6851-4faf-9438-e2e0d03ed79c
md"""
but conversion to a data frame or copying the column does not preserve the ordering.
"""

# ╔═╡ 83f17efc-3f4c-480d-a220-50db70aa9ff8
levels(DataFrame(tbl).Test)  # after conversion to a DataFrame the order is lost

# ╔═╡ da2209ab-537a-4c47-9723-5071f02b9af1
md"
To enforce an ordering and to relabel the tests in the plot, create a vector of pairs
"

# ╔═╡ 546419cf-14fc-4ec4-baca-a99922f0ce14
tlabels = [     # establish order and labels of tbl.Test
	"Run" => "Endurance",
	"Star_r" => "Coordination",
	"S20_r" => "Speed",
	"SLJ" => "Power Low",
	"BPT" => "Power Up",
]

# ╔═╡ 4cbf6ac4-ef96-43d5-a8ba-a9e2f3bc8bdc
md"""
Creating a summary of the mean score by `Test`, `Sex` and `age` (rounded to the nearest tenth of a year) is done using the split-apply-combine tools for data frames.
As part of this process the tests and sexes are relabelled.

The resulting summary table is split by `test`.
"""

# ╔═╡ 014617eb-12f7-4949-9995-cd100bec9b6c
sumTestSexAge1 = groupby(   # summary grouped data frame by test, sex and rounded age
	combine(
		groupby(
			select(DataFrame(tbl),
				:age => (x -> round.(x, digits=1)) => :age1,
				:Test => (t -> getindex.(Ref(Dict(tlabels)), t)) => :test,
				:Sex => (s -> ifelse.(s .== "female", "Girls", "Boys")) => :sex,
				:zScore,
			),
			[:age1, :sex, :test]),
		:zScore => mean,
		),
	:test,
)

# ╔═╡ 73455f7f-6897-451c-903a-0d1c391ccf40
begin
				# create the figure and panels (axes) within the figure
	fTest = Figure(resolution = (1000, 600))
	faxs =  [Axis(fTest[1, i]) for i in eachindex(tlabels)]
				# iterate over the test labels in the desired order
	for (i, lab) in enumerate(last.(tlabels))
					# create the label in a box at the top
		Box(fTest[1, i, Top()], backgroundcolor = :gray)
    	Label(fTest[1, i, Top()], lab, padding = (5, 5, 5, 5))
					# split the subdataframe by sex to plot the points
		for df in groupby(sumTestSexAge1[(test = lab,)], :sex)
			scatter!(
				faxs[i],
				df.age1,
				df.zScore_mean,
				color=ifelse(first(df.sex) == "Boys", :blue, :red),
				label=first(df.sex))
		end
	end
	
	axislegend(faxs[1], position = :lt)  # only one legend for point colors
	faxs[3].xlabel = "Age"               # only one axis label

	hideydecorations!.(faxs[2:end], grid = false)  # y labels on leftmost panel only
	linkaxes!(faxs...)                   # use the same axes throughout
	colgap!(fTest.layout, 10)            # tighten the spacing between panels
	
	fTest
end

# ╔═╡ Cell order:
# ╟─7f109de7-2565-4ab7-97a6-58c52f583257
# ╠═0c8c6b94-a059-11eb-196f-cde6617ea966
# ╟─e75bd3d6-c55b-4d30-903a-673cae83af2a
# ╠═379d2a3d-17ee-4173-b0f0-96ec639ec299
# ╟─68ef0bef-606f-41e8-b258-69a554fe6935
# ╠═c35830ca-891e-4ac2-85e0-9c1d386e39dd
# ╟─2fac75f6-e1cd-4727-b30d-c8e7952e8862
# ╠═9059f205-a459-422e-858d-0fbb103c8430
# ╟─2db1e041-6851-4faf-9438-e2e0d03ed79c
# ╠═83f17efc-3f4c-480d-a220-50db70aa9ff8
# ╟─da2209ab-537a-4c47-9723-5071f02b9af1
# ╟─546419cf-14fc-4ec4-baca-a99922f0ce14
# ╟─4cbf6ac4-ef96-43d5-a8ba-a9e2f3bc8bdc
# ╠═014617eb-12f7-4949-9995-cd100bec9b6c
# ╠═73455f7f-6897-451c-903a-0d1c391ccf40
