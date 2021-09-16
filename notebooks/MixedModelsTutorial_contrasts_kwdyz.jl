### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 7a3de02b-c423-496e-bf92-9981d71e5eab
begin
    using Arrow
	using Chain
	using DataFrames
	using MixedModels
	using StatsBase
	using StatsModels
	using StatsModels: ContrastsCoding
end

# ╔═╡ 09a33b05-568e-427a-bd66-812d271d1791
md"""
# Contrast Coding of Visual Attention Effects
## Date: 2021-05-29 (revised: 2021-09-16)
"""

# ╔═╡ 28c37807-b83f-4c3e-848d-a7401b4eda29
md"""
## Example data
We take the `KWDYZ` dataset ([Kliegl et al., 2011; Frontiers](https://doi.org/10.3389/fpsyg.2010.00238)). This is an experiment looking at three effects of visual cueing under four different cue-target relations (CTRs). Two horizontal rectangles are displayed above and below a central fixation point or they displayed in vertical orientation to the left and right of the fixation point.  Subjects react to the onset of a small visual target occuring at one of the four ends of the two rectangles. The target is cued validly on 70% of trials by a brief flash of the corner of the rectangle at which it appears; it is cued invalidly at the three other locations 10% of the trials each. 

We specify three contrasts for the four-level factor CTR that are derived from spatial, object-based, and attractor-like features of attention. They map onto sequential differences between appropriately ordered factor levels. Interestingly, a different theoretical perspective, derived from feature overlap, leads to a different set of contrasts. Can the results refute one of the theoretical perspectives?

We also have a dataset from a replication and extension of this study (Kliegl, Kuschela, & Laubrock, 2015). Both data sets are available in [R-package RePsychLing](https://github.com/dmbates/RePsychLing/tree/master/data/) (Baayen et al., 2014).

## Preprocessing
"""

# ╔═╡ ccf701bb-cf16-4339-bd93-64ffe87435bb
begin
	dat1 = DataFrame(Arrow.Table("./data/kwdyz11.arrow"))
	dat1 = select(dat1, :subj => :Subj, :tar => :CTR, :rt);

	# Descriptive statistics
	cellmeans = combine(groupby(dat1, [:CTR]), 
                             :rt => mean, :rt  => std, :rt => length, 
                             :rt => (x -> std(x)/sqrt(length(x))) => :rt_semean)
end

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

Going beyond the four level factor; it is also helpful to see the corresponding layout of the eight means for the interaction of A and B and C.

```
          C1              C2
      B1     B2        B1     B2
 A1   +1     -1   A1   -1     +1
 A2   -1     +1   A2   +1     -1
```

### A(2) x B(2) x C(3)

TO BE DONE
"""

# ╔═╡ a12c0a44-f02e-479e-94f9-c5d09c3b25ae
md"""
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MixedModels = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsModels = "3eaba693-59b7-5ba5-a881-562e759f1c8d"

[compat]
Arrow = "~1.6.2"
Chain = "~0.4.8"
DataFrames = "~1.2.2"
MixedModels = "~4.1.1"
StatsBase = "~0.33.10"
StatsModels = "~0.6.25"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arrow]]
deps = ["ArrowTypes", "BitIntegers", "CodecLz4", "CodecZstd", "DataAPI", "Dates", "Mmap", "PooledArrays", "SentinelArrays", "Tables", "TimeZones", "UUIDs"]
git-tree-sha1 = "b00e6eaba895683867728e73af78a00218f0db10"
uuid = "69666777-d1a9-59fb-9406-91d4454c9d45"
version = "1.6.2"

[[ArrowTypes]]
deps = ["UUIDs"]
git-tree-sha1 = "a0633b6d6efabf3f76dacd6eb1b3ec6c42ab0552"
uuid = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
version = "1.2.1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "42ac5e523869a84eac9669eaceed9e4aa0e1587b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.4"

[[BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "f50b5a99aa6ff9db7bf51255b5c21c8bc871ad54"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.2.5"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Chain]]
git-tree-sha1 = "cac464e71767e8a04ceee82a889ca56502795705"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.8"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "59fe0cb37784288d6b9f1baebddbf75457395d40"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.0"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "d19cd9ae79ef31774151637492291d75194fc5fa"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.7.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "c2dbc7e0495c3f956e4615b78d03c7aa10091d0c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.15"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7c365bdef6380b29cfc5caaf99688cd7489f9b87"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.2"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "b3e5984da3c6c95bcf6931760387ff2e64f508f3"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.1"

[[JSONSchema]]
deps = ["HTTP", "JSON", "ZipFile"]
git-tree-sha1 = "b84ab8139afde82c7c65ba2b792fe12e01dd7307"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.3"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "JSONSchema", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "575644e3c05b258250bb599e57cf73bbf1062901"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.9.22"

[[MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[MixedModels]]
deps = ["Arrow", "DataAPI", "Distributions", "GLM", "JSON3", "LazyArtifacts", "LinearAlgebra", "Markdown", "NLopt", "PooledArrays", "ProgressMeter", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StatsModels", "StructTypes", "Tables"]
git-tree-sha1 = "f318e42a48ec0a856292bafeec6b07aed3f6d600"
uuid = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
version = "4.1.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "748f6e1e4de814b101911e64cc12d83a6af66782"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.2"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "d80cb3327d1aeef0f59eacf225e000f86e4eee0a"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.3"

[[NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "2b597c46900f5f811bec31f0dcc88b45744a2a09"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.0+0"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "3fa15c1f8be168e76d59097f66970adc86bfeb95"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.25"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "8445bf99a36d703a09c601f9a57e2f83000ef2ae"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimeZones]]
deps = ["Dates", "Future", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "6c9040665b2da00d30143261aea22c7427aada1c"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.7"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─09a33b05-568e-427a-bd66-812d271d1791
# ╠═7a3de02b-c423-496e-bf92-9981d71e5eab
# ╟─28c37807-b83f-4c3e-848d-a7401b4eda29
# ╠═ccf701bb-cf16-4339-bd93-64ffe87435bb
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
# ╟─a12c0a44-f02e-479e-94f9-c5d09c3b25ae
# ╠═ab8b2f3a-c8d0-4f74-943b-9426d1c65deb
# ╟─aef896f9-83d6-45f7-ac28-ba7d1ab7064b
# ╠═1a9a46c9-42fb-4e27-87af-e9278547eae6
# ╟─93fcc1b6-98a8-41b5-a84d-46e5bd2df7b9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
