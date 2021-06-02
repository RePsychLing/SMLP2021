### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 814bea80-a2e7-11eb-1bf4-331f9343d0ea
begin
	using Pkg
	Pkg.activate(".")
	Pkg.add(["Arrow", "CairoMakie", "DataFrames", "PlutoUI"])
end

# ╔═╡ 74516f96-31f4-4c94-b963-fd4ed17eb758
using Arrow, CairoMakie, DataFrames, PlutoUI

# ╔═╡ 07f5c367-1be6-4708-9f5c-c23f646b229b
tbl = Arrow.Table("./data/fggk21.arrow")

# ╔═╡ ace577b2-2266-4238-ac3b-55128b2232c7
begin
	df = DataFrame(tbl)
	describe(df)
end

# ╔═╡ 449c4f12-6882-4cd8-86c4-c40cbd742fc8
md"## Raw score density"

# ╔═╡ 6a19454e-c0d3-4dd0-a7e7-4d8b7352f737
md"Test: $(@bind test Select(levels(tbl.Test)))  By Sex: $(@bind by_sex CheckBox())"

# ╔═╡ f09549bc-4a50-49ea-ba54-14bb1ad94a4b
begin
	fdensity = Figure(resolution = (1000, 500))
	axs = Axis(fdensity[1,1])
	tdf = filter(:Test => ==(test), df)
	if by_sex
		density!(
			axs,
			filter(:Sex => ==("female"), tdf).score,
			color=(:red, 0.1),
			label="Girls",
		)
		density!(
			axs,
			filter(:Sex => ==("male"), tdf).score,
			color=(:blue, 0.1),
			label="Boys",
		)
		axislegend(axs, position = :lt)
	else
		density!(axs, tdf.score)
	end
	fdensity
end		

# ╔═╡ Cell order:
# ╠═814bea80-a2e7-11eb-1bf4-331f9343d0ea
# ╠═74516f96-31f4-4c94-b963-fd4ed17eb758
# ╠═07f5c367-1be6-4708-9f5c-c23f646b229b
# ╟─ace577b2-2266-4238-ac3b-55128b2232c7
# ╟─449c4f12-6882-4cd8-86c4-c40cbd742fc8
# ╟─f09549bc-4a50-49ea-ba54-14bb1ad94a4b
# ╟─6a19454e-c0d3-4dd0-a7e7-4d8b7352f737
