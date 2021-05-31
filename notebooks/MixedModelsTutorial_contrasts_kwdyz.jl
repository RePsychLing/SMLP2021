### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 7719495e-c196-11eb-1d1e-dd64c3443737
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 7a3de02b-c423-496e-bf92-9981d71e5eab
begin
	using RCall, RData, Chain, DataFrames, CategoricalArrays
	using MixedModels, MixedModelsMakie

	using LinearAlgebra, Statistics
	using Statistics: mean, std

	using StatsModels
	using StatsModels: ContrastsCoding
end

# ╔═╡ 09a33b05-568e-427a-bd66-812d271d1791
md"""
# Contrast Coding of Visual Attention Effects
## Date: 2021-05-29
"""

# ╔═╡ 28c37807-b83f-4c3e-848d-a7401b4eda29
md"""
## Example data
We take the `KWDYZ` dataset (Kliegl et al., 2011; Frontiers). This is an experiment looking at three effects of visual cueing under four different cue-target relations (CTRs). Two horizontal rectangles are displayed above and below a central fixation point or they displayed in vertical orientation to the left and right of the fixation point.  Subjects react to the onset of a small visual target occuring at one of the four ends of the two rectangles. The target is cued validly on 70% of trials by a brief flash of the corner of the rectangle at which it appears; it is cued invalidly at the three other locations 10% of the trials each. 

We specify three contrasts for the four-level factor CTR that are derived from spatial, object-based, and attractor-like features of attention. They map onto sequential differences between appropriately ordered factor levels. Interestingly, a different theoretical perspective, derived from feature overlap, leads to a different set of contrasts. Can the results refute one of the theoretical perspectives?

We also have a dataset from a replication and extension of this study (Kliegl, Kuschela, & Laubrock, 2015). Both data sets are available in [R-package RePsychLing](https://github.com/dmbates/RePsychLing/tree/master/data/) (Baayen et al., 2014).

## Preprocessing
"""

# ╔═╡ 8ec41516-4609-489e-9a84-f5f5a2d9fc51
begin
	dat1 = @chain begin
		only(values(load("./data/KWDYZ.rda")))     # .rda -> Dict{String,Any}
		select(:subj => :Subj, :tar => :CTR, :rt)
        transform(:CTR => (v -> levels!(v, ["val", "sod", "dos", "dod"])) => :CTR)
	end
	
	# Descriptive statistics
	cellmeans = combine(groupby(dat1, [:CTR]), 
                             :rt => mean, :rt  => std, :rt => length, 
                             :rt => (x -> std(x)/sqrt(length(x))) => :rt_semean)


	OM, GM = mean(dat1.rt), mean(cellmeans.rt_mean)
end

# ╔═╡ d5a246f0-b172-4a67-828a-74fd58853de5
cellmeans

# ╔═╡ cb63bd4b-5f0e-49d1-a257-2b83c1687975
md"""## SeqDiffCoding 
This contrast corresponds to `MASS::contr.sdif()` in R.
"""

# ╔═╡ 8ffba70f-fc97-47a2-b973-3c0f2047eb9b
begin
	cntr1 = Dict(
    :CTR  => SeqDiffCoding(levels=["val", "sod", "dos", "dod"]),
    :Subj => Grouping()
	);


	formula = @formula  rt ~ 1 + CTR + (1 + CTR | Subj);
	m1 = fit(MixedModel, formula, dat1, contrasts=cntr1)
end

# ╔═╡ 4e9359a3-2533-4ce8-b2fd-26411e84f59c
md"""
## HypothesisCoding 
This contrast corresponds to `MASS::contr.sdif()` in R.
A general solution (not inverse of last contrast) 
"""

# ╔═╡ b00e953a-b279-4ec3-8744-5518d1242d8e
begin
	cntr1b = Dict(
    :CTR => HypothesisCoding([-1  1  0  0
                               0 -1  1  0
                               0  0  1 -1],
            levels=["val", "sod",  "dos", "dod"],
            labels=["spt", "obj", "grv"])
	);

	m1b = fit(MixedModel, formula, dat1, contrasts=cntr1b)
end

# ╔═╡ 7a3749b3-d2e9-4b53-9e57-975ee428b416
md"""

Controlling the ordering of levels for contrasts:

1.  kwarg `levels=` to order the levels; the first is set as the baseline.
2.  kwarg `base=` to fix the baseline level.

The assignment of random factors such as `Subj` to `Grouping()` is only necessary when the sample size is very large and leads to an out-of-memory error; it is included only in the first example for reference.

## DummyCoding 
Thi contrast corresponds to `contr.treatment()` in R
"""

# ╔═╡ 6393d21d-a3a6-4463-8981-052440ba7887
begin
	cntr2 = Dict(:CTR => DummyCoding(base= "val"));

	m2 = fit(MixedModel, formula, dat1, contrasts=cntr2)
end

# ╔═╡ d087afe8-9d2b-489e-9160-206f22486505
md"""
This contrast has the disadvantage that the intercept returns the mean of the level specified as `base`, default is the first level, not the GM. 

## YchycaeitCoding

The contrasts returned by `DummyCoding` may be what you want. Can't we have them, but also the GM rather than the mean of the base level?  Yes, we can!  I call this "You can have your cake and it eat, too"-Coding (YchycaeitCoding). 
"""

# ╔═╡ bb97e862-24ad-4ce5-9011-429a0e69e5ab
begin
	cntr2b = Dict(
    :CTR => HypothesisCoding([-1  1  0  0
                              -1  0  1  0
                              -1  0  0  1],
            levels=["val", "sod",  "dos", "dod"])
	);

	m2b = fit(MixedModel, formula, dat1, contrasts=cntr2b)
end

# ╔═╡ 71528268-41fe-4de5-8725-1c65c4839c65
md"""
Just relevel the factor or move the column with -1s for a different base.

## EffectsCoding - corresponds to `contr.sum()` in R
"""

# ╔═╡ e2372cb1-03d4-4404-bfef-640ad5653792
begin
	cntr3 = Dict(:CTR => EffectsCoding(base= "dod"));

	m3 = fit(MixedModel, formula, dat1, contrasts=cntr3)
end

# ╔═╡ 142c86d3-e340-494c-8b52-8f2013993a19
md"## HelmertCoding"

# ╔═╡ 0bb71abe-433c-44e3-bff4-eb26e63bee2d
begin
	cntr4 = Dict(:CTR => HelmertCoding());

	fit(MixedModel, formula, dat1, contrasts=cntr4)
end

# ╔═╡ 0cf43493-af05-4744-a154-d7209251ae8f
md"## Reverse HelmertCoding"

# ╔═╡ 2d778d8e-949c-4a7b-a774-770c6bc4bce6
begin
	cntr4b = Dict(:CTR => HelmertCoding(levels=reverse(levels(dat1.CTR))));

	fit(MixedModel, formula, dat1, contrasts=cntr4b)
end

# ╔═╡ a71103ca-a680-45ac-ba02-61a5660cc157
md"""
Helmert contrasts are othogonal.

## AnovaCoding - Anova contrasts are orthogonal.

### A(2) x B(2)

An A(2) x B(2) design can be recast as an F(4) design with the levels (A1-B1, A1-B2, A2-B1, A2-B2). The following contrast specifiction returns estimates for the main effect of A, the main effect of B, and the interaction of A and B. In a figure With A on the x-axis and the levels of B shown as two lines, the interaction tests the null hypothesis that the two lines are parallel. A positive coefficient implies overadditivity (diverging lines toward the right) and a negative coefficient underadditivity (converging lines).
"""

# ╔═╡ 00bedbae-0bc4-4e4e-85d5-91ffc8ea77ee
begin	
	cntr5 = Dict(
    :CTR => HypothesisCoding([-1  -1 +1  +1          # A
                              -1  +1 -1  +1          # B
                              +1  -1 -1  +1],        # A x B
            levels=["val", "sod",  "dos", "dod"],
            labels=["A", "B", "AxB"])
	);
	m5 = fit(MixedModel, formula, dat1, contrasts=cntr5)
end

# ╔═╡ f504edce-35de-4938-a3c6-08d4fc8f2e66
md"""
It is also helpful to see the corresponding layout of the four means for the interaction of A and B (i.e., the third contrast)

```
        B1     B2
   A1   +1     -1
   A2   -1     +1
```

Thus, interaction tests whether the difference between main diagonal and minor diagonal is different from zero. 
=#

### A(2) x B(2) x C(2)

Going beyond the four level factor. You can obtain this information with a few lines of code. 
"""

# ╔═╡ 6e13cb55-fc02-436d-956e-42b40f46521e
begin	
	R"""
	df1 <- expand.grid(C=c("c1","c2"), B=c("b1","b2"), A = c("a1", "a2"))
	contrasts(df1$A) <- contrasts(df1$B) <- contrasts(df1$C) <- contr.sum(2)
	X <- model.matrix( ~ A*B*C, df1) # get design matrix for experiment
	X_1 <- t(X[,2:8]) # transponse; we don't need the intercept column
	"""

	X_1 = @rget X_1;
	cntr6 = Dict(:CTR => HypothesisCoding(X_1));
end

# ╔═╡ 4ba345a7-adf8-4b74-9ca3-1467f7a693b9
md"""
If you want subtract level 1 from level 2, multiply X with -1. I did not check whether the assignment of `X_1` in the last line really works. In any case you can use `X_1` as a starting point for explicit specification with `HypothesisCoding`.

It is also helpful to see the corresponding layout of the four means for the interaction of A and B (i.e., the third contrast)

```
          C1              C2
      B1     B2        B1     B2
 A1   +1     -1   A1   -1     +1
 A2   -1     +1   A2   +1     -1
```

### A(2) x B(2) x C(3)

For the three-level factor C we need to decide on a contrast. I am using the orthogonal Helmert contrast, because for non-orthogonal contrasts the correspondence between R and Julia breaks down. This is not a show-stopper, but requires a bit more coding. It is better to implement the following R chunk in Julia. 
"""

# ╔═╡ d5649c0b-b49a-4d9e-9302-fd191820681c
begin
	R"""
	df2 <- expand.grid( C=c("c1","c2","c3"),  B=c("b1","b2"), A = c("a1", "a2"))
	contrasts(df2$A) <- contrasts(df2$B) <- contr.sum(2)
	contrasts(df2$C) <- contr.helmert(3)

	X <- model.matrix( ~ A*B*C, df2) # get design matrix for experiment
	X_2 <- MASS::fractions(t(X[,2:12])) # transpose; we don't need the intercept 		column
"""

X_2 = @rget X_2;
cntr7 = Dict(:CTR => HypothesisCoding(X_2))
end

# ╔═╡ a12c0a44-f02e-479e-94f9-c5d09c3b25ae
md"""
I have not checked whether the assignment of `X_2` really works. In any case you can use `X_2`as a starting point for explicit specification with `HypothesisCoding`.


## NestedCoding

An A(2) x B(2) design can be recast as an F(4) design with the levels (A1-B1, A1-B2, A2-B1, A2-B2).  The following contrast specifiction returns an estimate for the main effect of A and the effects of B nested in the two levels of A. In a figure With A on the x-axis and the levels of B shown as two lines, the second contrast tests whether A1-B1 is different from A1-B2 and the third contrast tests whether A2-B1 is different from A2-B2.
"""

# ╔═╡ ab8b2f3a-c8d0-4f74-943b-9426d1c65deb
begin
	cntr8 = Dict(
    :CTR => HypothesisCoding([-1  -1 +1  +1          
                              -1  +1  0   0
                               0   0 +1  -1],
            levels=["val", "sod",  "dos", "dod"],
            labels=["do_so", "spt", "grv"])
	);
	m8 = fit(MixedModel, formula, dat1, contrasts=cntr8)
end

# ╔═╡ aef896f9-83d6-45f7-ac28-ba7d1ab7064b
md"""
The three contrasts for one main effect and two nested contrasts are orthogonal. There is no test of the interaction (parallelism).

## Other orthogonal contrasts

For factors with more than four levels there are many options for specifying orthogonal contrasts as long as one proceeds in a top-down strictly hiearchical fashion. 

Suppose you have a factor with seven levels and let's ignore shifting colummns. In this case, you have six options for the first contrast, that is 6 vs. 1, 5 vs.2 , 4 vs. 3, 3 vs. 4, 2 vs. 5, and 1 vs. 6 levels.  Then, you specify orthogonal contrasts for partitions with more than 2 elements and so on. That is, you don't specify a contrast that crosses an earlier partition line.  

In the following example, after an initial 4 vs 3 partitioning of levels, we specify `AnovaCoding` for the left and `HelmertCoding` for the right partition.
"""

# ╔═╡ 1a9a46c9-42fb-4e27-87af-e9278547eae6
begin
	cntr9 = Dict(
    :CTR => HypothesisCoding(
    [-1/4 -1/4 -1/4 -1/4 +1/3 +1/3 +1/3          
     -1/2 -1/2 +1/2 +1/2   0    0    0
     -1/2 +1/2 -1/2 +1/2   0    0    0 
     +1/2 -1/2 -1/2 +1/2   0    0    0
       0    0    0    0   -1   +1    0
       0    0    0    0  -1/2 -1/2   1
     ],
    levels=["A1", "A2",  "A3", "A4", "A5", "A6", "A7"],
    labels=["c567.1234", "B", "C", "BxC", "c6.5", "c6.56"])
	)
end

# ╔═╡ 93fcc1b6-98a8-41b5-a84d-46e5bd2df7b9
md"""
There are two rules that hold for all orthogonal contrasts:

1. The weights within rows sum to zero.
2. For all pairs of rows, the sum of the products of weights in the same columns sums to zero. 

# Appendix: Summary (Dave Kleinschmidt)

[StatsModels](https://juliastats.org/StatsModels.jl/latest/contrasts/)

StatsModels.jl provides a few commonly used contrast coding schemes,
some less-commonly used schemes, and structs that allow you to manually
specify your own, custom schemes. 

## Standard contrasts

The most commonly used contrasts are `DummyCoding` and `EffectsCoding`
(which are similar to `contr.treatment()` and `contr.sum()` in R,
respectively).

## "Exotic" contrasts (rk: well ...)

We also provide `HelmertCoding` and `SeqDiffCoding` (corresponding to
base R's `contr.helmert()` and `MASS::contr.sdif()`).

## Manual contrasts

**ContrastsCoding()**

There are two ways to manually specify contrasts. First, you can specify
them **directly** via `ContrastsCoding`. If you do, it's good practice
to specify the levels corresponding to the rows of the matrix, although
they can be omitted in which case they'll be inferred from the data.

**HypothesisCoding()**

A better way to specify manual contrasts is via `HypothesisCoding`, where each
row of the matrix corresponds to the weights given to the cell means of the
levels corresponding to each column (see [Schad et
al. 2020](https://doi.org/10.1016/j.jml.2019.104038) for more information). 
"""

# ╔═╡ Cell order:
# ╟─7719495e-c196-11eb-1d1e-dd64c3443737
# ╟─09a33b05-568e-427a-bd66-812d271d1791
# ╟─7a3de02b-c423-496e-bf92-9981d71e5eab
# ╟─28c37807-b83f-4c3e-848d-a7401b4eda29
# ╠═8ec41516-4609-489e-9a84-f5f5a2d9fc51
# ╠═d5a246f0-b172-4a67-828a-74fd58853de5
# ╟─cb63bd4b-5f0e-49d1-a257-2b83c1687975
# ╠═8ffba70f-fc97-47a2-b973-3c0f2047eb9b
# ╟─4e9359a3-2533-4ce8-b2fd-26411e84f59c
# ╠═b00e953a-b279-4ec3-8744-5518d1242d8e
# ╟─7a3749b3-d2e9-4b53-9e57-975ee428b416
# ╠═6393d21d-a3a6-4463-8981-052440ba7887
# ╟─d087afe8-9d2b-489e-9160-206f22486505
# ╠═bb97e862-24ad-4ce5-9011-429a0e69e5ab
# ╟─71528268-41fe-4de5-8725-1c65c4839c65
# ╠═e2372cb1-03d4-4404-bfef-640ad5653792
# ╟─142c86d3-e340-494c-8b52-8f2013993a19
# ╠═0bb71abe-433c-44e3-bff4-eb26e63bee2d
# ╟─0cf43493-af05-4744-a154-d7209251ae8f
# ╠═2d778d8e-949c-4a7b-a774-770c6bc4bce6
# ╟─a71103ca-a680-45ac-ba02-61a5660cc157
# ╠═00bedbae-0bc4-4e4e-85d5-91ffc8ea77ee
# ╟─f504edce-35de-4938-a3c6-08d4fc8f2e66
# ╠═6e13cb55-fc02-436d-956e-42b40f46521e
# ╟─4ba345a7-adf8-4b74-9ca3-1467f7a693b9
# ╠═d5649c0b-b49a-4d9e-9302-fd191820681c
# ╟─a12c0a44-f02e-479e-94f9-c5d09c3b25ae
# ╠═ab8b2f3a-c8d0-4f74-943b-9426d1c65deb
# ╟─aef896f9-83d6-45f7-ac28-ba7d1ab7064b
# ╠═1a9a46c9-42fb-4e27-87af-e9278547eae6
# ╟─93fcc1b6-98a8-41b5-a84d-46e5bd2df7b9
