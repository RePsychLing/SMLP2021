### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 9396fcac-b0b6-11eb-3a60-9f2ce25df953
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 4ecf1267-aac9-4e8f-a522-5595b1205f43
begin
	using CSV, RCall, DataFrames, MixedModels, MixedModelsMakie
	using LinearAlgebra, Statistics, StatsBase
	using CategoricalArrays, Arrow
	using AlgebraOfGraphics
	using AlgebraOfGraphics: linear
	using CairoMakie  # for Scatter
end

# ╔═╡ eebef932-1d63-41e5-998f-9748379c43af
md"""
# Mixed Models Tutorial: Conditional Modes and Caterpillar Plots

This script uses a subset of data reported in Fühner, Golle, Granacher, & Kliegl (2021). 
Physical fitness in third grade of primary school: 
A mixed model analysis of 108,295 children and 515 schools.

All children were between 6.0 and 6.99 years at legal keydate (30 September) of school enrollement, 
that is in their ninth year of life in the third grade. To avoid delays associated with model 
fitting we work with a reduced data set and less complex models than those in the reference 
publication. The script requires only a few changes to specify the more complex models in the paper. 

The script is structured in three main sections and an Appendix: 

1. **Setup** with reading and examing the data
2. **Extraction of conditional modes** 
3. **Caterpillar plot**
4. **Appendix:** Soruce of MixedModelsMakie `caterpillar()` function

## 1. Setup
### 1.0 Packages and functions
"""

# ╔═╡ faafd359-fb38-4f49-9dfd-19cb4f0c5a54
begin
	function viewdf(x)
		DataFrame(Arrow.Table(take!(Arrow.write(IOBuffer(), x))))
	end
end

# ╔═╡ ee85abd9-0172-44d3-b03a-c6a780d33c72
md"""
### 1.1 Readme for 'EmotikonSubset.rds'

1. Cohort: 9 levels; 2011-2019
2. School: 46 levels 
3. Child: 11,566 levels
4. Sex: 5,893 girls, 5,673 boys
5. age: test date - middle of month of birthdate (ranges between 7.9 and 9.2)
6. Test: 5 levels
     + Endurance (`Run`):  6 minute endurance run [m]; to nearest 9m in 9x18m field
     + Coordination (`Star_r`): star coordination run [m/s]; 9x9m field, 4 x diagonal = 50.912 m
     + Speed(`S20_r`): 20-meters sprint [m/s]
     + Muscle power low (`SLJ`): standing long jump [cm] 
     + Muscle power up (`BPT`): 1-kg medicine ball push test [m] 
 7. score - see units

### 1.2 Preprocessing
#### Read data
"""

# ╔═╡ 39443fb0-64ec-4a87-b072-bc8ad6fa9cf4
dat = rcopy(R"readRDS('./data/EmotikonSubset.rds')");

# ╔═╡ 0051ec83-7c30-4d28-9dfa-4c6f5d0259fa
md"""
#### Transformations
"""

# ╔═╡ 7440c439-9d4c-494f-9ac6-18c9fb2fe144
begin
	transform!(dat, :age, :age => (x -> x .- 8.5) => :a1); # centered age (linear)
	transform!(dat,  :a1, :a1  => (x -> x.^2) => :a2);     # centered age (quadr.)
	select!(groupby(dat,  :Test), :, :score => zscore => :zScore); # z-score
end;

# ╔═╡ 16c3cdaa-49fa-46d5-a844-03094329fe4c
viewdf(dat)

# ╔═╡ 64cc1f8e-f831-4a53-976f-dc7600b5634d
# ... by Test and Sex
begin
	dat2 = combine(groupby(dat, [:Test, :Sex]), 
                             :score => mean, :score  => std, 
                             :zScore => mean, :zScore => std)
	viewdf(dat2)
end

# ╔═╡ c6c6f056-b8b9-4190-ac14-b900bafa04df
md"""
## 2 LMMs and extraction of conditional modes 
### 2.1 _SeqDiffCoding_ of `Test`

_SeqDiffCoding_ was used in the publication. This specification tests pairwise 
differences between the five neighboring levels of `Test`, that is: 

+ H1: `Star_r` - `Run` (2-1)
+ H2: `S20_r` - `Star_r` (3-2) 
+ H3: `SLJ` - `S20_r` (4-3)
+ H4: `BPT` - `SLJ` (5-4)

Various options for contrast coding are the topic of the *MixedModelsTutorial_contrasts.jl*
notbebook.
"""

# ╔═╡ c5326753-a03b-4739-a82e-90ffa7c1ebdb
begin
	contr = merge(
        Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		Dict(:Sex => EffectsCoding(; levels=["Girls", "Boys"])),
	    Dict(:Test => SeqDiffCoding(; levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"])),
        Dict(:TestHC => HelmertCoding(; levels=["S20_r", "SLJ", "Star_r", "Run", "BPT"])),
	   )
end;

# ╔═╡ e3cf1ee6-b64d-4356-af23-1dd5c7a0fec6
md"""
#### 2.2 LMMs  `m11` and `m12` for the reduced data

LMM `m11` corresponds to LMM `m1` in the publication, but fit to the reduced set of data. Conditional (co-)variances cannot not (yet) be computed with `Child` in the LMM. Therefore, we estimate also LMM `m12` without the random factor `Child`. For the final LMM we also estimate conditional covariances for scores, rather than effects.

"""

# ╔═╡ 0dd0d060-81b4-4cc0-b306-dda7589adbd2
begin
    f11 = @formula zScore ~ 1 + Test * a1 * Sex +  
                           (1 + Test + a1 | School) + (1 + Test | Child) +
                   zerocorr(1 + Test | Cohort);
    m11 = fit(MixedModel, f11, dat)
    VarCorr(m11)
end

# ╔═╡ f68f6884-7c37-4868-b0b7-01ad450ee383
begin
    f12 = @formula zScore ~ 1 + Test * a1 * Sex +  
                           (1 + Test + a1 | School) + 
                   zerocorr(1 + Test | Cohort);
    m12 = fit(MixedModel, f12, dat)
    VarCorr(m12)
end

# ╔═╡ a1186b62-81fd-4f99-ba59-45c213ec264b
begin
    f13 = @formula zScore ~ 1 + Test * a1 * Sex +  
                           (0 + Test + a1 | School) + 
                   zerocorr(0 + Test | Cohort);
    m13 = fit(MixedModel, f13, dat)
    VarCorr(m13)
end

# ╔═╡ ef1dfbf4-700c-4093-9824-01368c511531
md"""
### 2.2 Conditional modes (BLUPs)
#### Three random factors
"""

# ╔═╡ ace9a2f8-075a-49b5-94eb-12be4ffb6568
begin
 cms11 = raneftables(m11);   
 cms11_Schl = DataFrame(cms11.School);
 cms11_Chld = DataFrame(cms11.Child);
 cms11_Chrt = DataFrame(cms11.Cohort)
end

# ╔═╡ 167fbd71-2a7b-4afa-8ffb-815dcda63129
md"""
#### Two random factors -- w/o Child

First LMM m12 with effects; the second LMM m13 with scores.
"""

# ╔═╡ 68ee8abb-04e6-45ee-8f0a-033e7d2ef57b
begin
  cms12 = raneftables(m12);   
  cms12_Schl = DataFrame(cms12.School);
  cms12_Chrt = DataFrame(cms12.Cohort);
end

# ╔═╡ 3a3c50c6-7372-49f9-94c9-36b4d71d5ff6
begin
  cms13 = raneftables(m13);   
  cms13_Schl = DataFrame(cms13.School);
  cms13_Chrt = DataFrame(cms13.Cohort);
end

# ╔═╡ dd8b5c05-c1c8-4778-b179-1ad39f50f8d3
md"""
### 2.3 Conditional (co-)variances
### Three random factors
"""

# ╔═╡ 25376bb3-4673-4c89-8600-6e3da85971b4
BlockDescription(m11)

# ╔═╡ e7cdd374-57c3-4d3c-9b1b-b6a211733b81
cvs11 = condVar(m11) 

# ╔═╡ 1a2586a6-ff9d-4067-a5af-6c0ee6929e81
md"""
### Two random factors - w/o Child
First LMM m12 with effects; then LMM m13 with scores.
"""

# ╔═╡ d2c0979e-ed52-4b56-b3ec-58db1609843e
begin
    BlockDescription(m12)
    cvs12 = condVar(m12);    # 2-element Vector{Array{Float64, 3}}
	cvs12_Schl = cvs12[1] 
end

# ╔═╡ cf3f5312-6702-4069-95aa-862dc54d9a34
cvs12_Chrt = cvs12[2]    # 5×5×9 Array{Float64, 3}:

# ╔═╡ 4c5f8c2a-85b0-45a6-925d-1deaa1f302d2
md"""
We can also look at scores rather than effects.
"""

# ╔═╡ 17d31da0-9579-4d8f-9b7f-2cd987200d2e
begin
    BlockDescription(m13)
    cvs13 = condVar(m13);    # 2-element Vector{Array{Float64, 3}}
	cvs13_Schl = cvs13[1] 
end

# ╔═╡ 90ffbb02-139c-4a28-ac76-47e414974840
cvs13_Chrt = cvs13[2] 


# ╔═╡ 5a62529e-cc10-4e2c-8d75-125951f2dc34
md"""
## 3 Caterpillar plots

A "caterpillar plot" is a horizontal error-bar plot of conditional modes and credibility intervals.

    caterpillar(m::LinearMixedModel, gf::Symbol)

In the case f LMMs, conditonal modes are the conditional means.
"""

# ╔═╡ cb6195ea-296d-4cd2-9be0-906ebf42d5a3
md"""
### 3.1 LMM `m11_Cohort`
#### **Effects**
"""

# ╔═╡ d9459ffb-f9db-4a4e-9e25-5b34df6540c5
md"""
### 3.2 LMM `m12_Cohort`
#### **Effects**
"""

# ╔═╡ 58651ebc-2c69-4814-af5e-87fd5d7c3b52
md"""#### Scores"""

# ╔═╡ 53ed436c-82b9-4331-bef4-0e034c233e7f
md"""
### 3.3 LMM `m12_School`
#### Effects
"""

# ╔═╡ 61cd093e-aa4e-475a-9d62-3ddd3e2b5fd7
md"""
#### Scores
"""

# ╔═╡ 10ef55d6-6048-4353-92e5-e4621bfb130e
md"""
## 4 APPENDIX: Source of MixedModelsMakie caterpillar function

These functions are **disabled** at the start of the notebook. The `caterpillar` function is available from the package `MixedModelsMakie`.  It generates and uses information on random effects conditional modes/means, variances, etc.

"""

# ╔═╡ cbe2197e-12c7-4d55-8817-d25d7b7cc5bd
md"""

struct RanefInfo{T<:AbstractFloat}
    cnames::Vector{String}
    levels::Vector
    ranef::Matrix{T}
    stddev::Matrix{T}
end

"""

# ╔═╡ 32acfdf0-ecb6-4acb-904f-d7b963771d6d
md"""

`ranefinfo(m::LinearMixedModel)`

Return a `NamedTuple{fnames(m), NTuple(k, RanefInfo)}` from model `m`
"""

# ╔═╡ 90ce17cc-d497-4605-aa62-be69343e03eb
function ranefinfo(m::LinearMixedModel{T}) where {T}
    fn = fnames(m)
    val = sizehint!(RanefInfo[], length(fn))
    for (re, eff, cv) in zip(m.reterms, ranef(m), condVar(m))
        push!(
            val, 
            RanefInfo(
                re.cnames,
                re.levels,
                Matrix(adjoint(eff)),
                Matrix(adjoint(dropdims(sqrt.(mapslices(diag, cv, dims=1:2)), dims=2))),
            )
        )
    end
    NamedTuple{fn}((val...,))
end

# ╔═╡ ffcc35da-25b8-4bee-80d2-93b033584cd4
md"""
  
`caterpillar!(f::Figure, r::RanefInfo; orderby=1)`

Add Axes of a caterpillar plot from `r` to `f`.

The order of the levels on the vertical axes is increasing `orderby` column
of `r.ranef`, usually the `(Intercept)` random effects.
"""


# ╔═╡ d1efb45b-44e4-4a6c-a967-0dc054a5ae83
function caterpillar!(f::Figure, r::RanefInfo; orderby=1)
    rr = r.ranef
    vv = view(rr, :, orderby)
    ord = sortperm(vv)
    y = axes(rr, 1)
    cn = r.cnames
    axs = [Axis(f[1, j]) for j in axes(rr, 2)]
    linkyaxes!(axs...)
    for (j, ax) in enumerate(axs)
        xvals = view(rr, ord, j)
        scatter!(ax, xvals, y, color=(:red, 0.2))
        errorbars!(ax, xvals, y, view(r.stddev, ord, j), direction=:x)
        ax.xlabel = cn[j]
        ax.yticks = y
        j > 1 && hideydecorations!(ax, grid=false)
    end
    axs[1].yticks = (y, r.levels[ord])
    f
end


# ╔═╡ 64e3f006-9120-40a5-97da-8d6694a7cc87
begin
	re11_Chrt = ranefinfo(m11)[:Cohort];
	caterpillar!(Figure(; resolution=(800,600)), re11_Chrt; orderby=1)
end

# ╔═╡ 9a0194cd-3f96-4014-b146-a7842a4fa22f
begin
	re12_Chrt = ranefinfo(m12)[:Cohort];
	caterpillar!(Figure(; resolution=(800,600)), re12_Chrt; orderby=1)
end

# ╔═╡ 407dcab5-40e4-4b53-a5ce-b771c76b8497
begin
	re13_Chrt = ranefinfo(m13)[:Cohort];
	caterpillar!(Figure(; resolution=(800,600)), re13_Chrt; orderby=2)
end

# ╔═╡ 26465af0-c83b-4e1c-9a8d-800a8140e5c0
begin
	re12_Schl = ranefinfo(m12)[:School];
	caterpillar!(Figure(; resolution=(800,600)), re12_Schl; orderby=3)
end

# ╔═╡ 205b7fdb-2db6-4c81-a75b-b0729f57735e
begin
	re13_Schl = ranefinfo(m13)[:School];
	caterpillar!(Figure(; resolution=(800,600)), re13_Schl; orderby=6)
end

# ╔═╡ 781486fc-79b4-4e2e-9902-c29c3e4460b6
md"""
    caterpillar(m::LinearMixedModel, gf::Symbol)
Returns a `Figure` of a "caterpillar plot" of the random-effects means and prediction intervals
A "caterpillar plot" is a horizontal error-bar plot of conditional means and standard deviations
of the random effects.
"""

# ╔═╡ 28e8dc7f-7841-4651-9119-de785086124a
function caterpillar(m::LinearMixedModel, gf::Symbol=first(fnames(m)))
    caterpillar!(Figure(resolution=(1000,800)), ranefinfo(m)[gf])
end

# ╔═╡ Cell order:
# ╠═9396fcac-b0b6-11eb-3a60-9f2ce25df953
# ╟─eebef932-1d63-41e5-998f-9748379c43af
# ╠═4ecf1267-aac9-4e8f-a522-5595b1205f43
# ╠═faafd359-fb38-4f49-9dfd-19cb4f0c5a54
# ╟─ee85abd9-0172-44d3-b03a-c6a780d33c72
# ╠═39443fb0-64ec-4a87-b072-bc8ad6fa9cf4
# ╟─0051ec83-7c30-4d28-9dfa-4c6f5d0259fa
# ╠═7440c439-9d4c-494f-9ac6-18c9fb2fe144
# ╠═16c3cdaa-49fa-46d5-a844-03094329fe4c
# ╠═64cc1f8e-f831-4a53-976f-dc7600b5634d
# ╟─c6c6f056-b8b9-4190-ac14-b900bafa04df
# ╠═c5326753-a03b-4739-a82e-90ffa7c1ebdb
# ╟─e3cf1ee6-b64d-4356-af23-1dd5c7a0fec6
# ╠═0dd0d060-81b4-4cc0-b306-dda7589adbd2
# ╠═f68f6884-7c37-4868-b0b7-01ad450ee383
# ╠═a1186b62-81fd-4f99-ba59-45c213ec264b
# ╟─ef1dfbf4-700c-4093-9824-01368c511531
# ╠═ace9a2f8-075a-49b5-94eb-12be4ffb6568
# ╟─167fbd71-2a7b-4afa-8ffb-815dcda63129
# ╠═68ee8abb-04e6-45ee-8f0a-033e7d2ef57b
# ╠═3a3c50c6-7372-49f9-94c9-36b4d71d5ff6
# ╠═dd8b5c05-c1c8-4778-b179-1ad39f50f8d3
# ╠═25376bb3-4673-4c89-8600-6e3da85971b4
# ╠═e7cdd374-57c3-4d3c-9b1b-b6a211733b81
# ╟─1a2586a6-ff9d-4067-a5af-6c0ee6929e81
# ╠═d2c0979e-ed52-4b56-b3ec-58db1609843e
# ╠═cf3f5312-6702-4069-95aa-862dc54d9a34
# ╠═4c5f8c2a-85b0-45a6-925d-1deaa1f302d2
# ╠═17d31da0-9579-4d8f-9b7f-2cd987200d2e
# ╠═90ffbb02-139c-4a28-ac76-47e414974840
# ╟─5a62529e-cc10-4e2c-8d75-125951f2dc34
# ╠═cb6195ea-296d-4cd2-9be0-906ebf42d5a3
# ╠═64e3f006-9120-40a5-97da-8d6694a7cc87
# ╠═d9459ffb-f9db-4a4e-9e25-5b34df6540c5
# ╠═9a0194cd-3f96-4014-b146-a7842a4fa22f
# ╟─58651ebc-2c69-4814-af5e-87fd5d7c3b52
# ╠═407dcab5-40e4-4b53-a5ce-b771c76b8497
# ╠═53ed436c-82b9-4331-bef4-0e034c233e7f
# ╠═26465af0-c83b-4e1c-9a8d-800a8140e5c0
# ╟─61cd093e-aa4e-475a-9d62-3ddd3e2b5fd7
# ╠═205b7fdb-2db6-4c81-a75b-b0729f57735e
# ╟─10ef55d6-6048-4353-92e5-e4621bfb130e
# ╠═cbe2197e-12c7-4d55-8817-d25d7b7cc5bd
# ╟─32acfdf0-ecb6-4acb-904f-d7b963771d6d
# ╠═90ce17cc-d497-4605-aa62-be69343e03eb
# ╟─ffcc35da-25b8-4bee-80d2-93b033584cd4
# ╠═d1efb45b-44e4-4a6c-a967-0dc054a5ae83
# ╠═781486fc-79b4-4e2e-9902-c29c3e4460b6
# ╠═28e8dc7f-7841-4651-9119-de785086124a
