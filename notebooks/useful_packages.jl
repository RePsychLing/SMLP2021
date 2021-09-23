### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 539bb62b-e4e3-4d72-837c-302ac9335e2e
begin
	using PlutoUI
	TableOfContents()
end

# ╔═╡ 27130015-4f45-4ebd-9094-07ec00261e5f
md"""
# Useful Packages

Unlike R, Julia does not immediately expose a huge number of functions, but instead requires loading packages (whether from the standard library or from the broader package ecosystem) for a lot of relevant functionality for statistical analysis. There are technical reasons for this, but one further motivation is that Julia is at a broader "technical computing" audience (like MATLAB or perhaps Python) and less at a "statistical analysis" audience. 

This has two important implications:
1. Even relatively simple programs will often load several packages.
2. Packages are often focused on adding a relatively narrow set of functionality, which means that "basic" functionality (e.g. reading a CSV file and manipuating it as a DataFrame) is often split across multiple packages. In other words, **see the the first point!**

This notebook is not intended to be an exhaustive list of packages, but rather to highlight a few packages that I suspect will be particularly useful.
Before getting onto the packages, I have one final hint: take advantage of how easy and first-class package management in Julia is. Having good package management makes reproducible analyses much easier and avoids breaking old analyses when you start a new one. Pluto helpfully installs and manages for you, but the package-manager REPL mode (activated by typing `]` at the `julia>` prompt) is very useful.
"""

# ╔═╡ cd73a70f-4ce0-4ef8-87be-d3b319066602
md"""

## Data wrangling

### Reading data

- [Arrow.jl](https://arrow.juliadata.org/dev/manual/) a high performance format for data storage, accessible in R via the [`arrow` package](https://arrow.apache.org/docs/r/) and in Python via `pyarrow`. (Confusingly, the function for reading and writing Arrow format files in R is called `read_feather` and `write_feather`, but the modern Arrow format is distinct from the older Feather format provided by the `feather` package.) This is the format that we store the example and test datasets in for MixedModels.jl.

- [CSV.jl](https://csv.juliadata.org/stable/index.html) useful for reading comma-separated values, tab-separated values and basically everything handled by the `read.csv` and `read.table` family of functions in R. 

Note that by default both Arrow.jl and CSV.jl do not return a DataFrame, but rather "column tables" -- named tuples of column vectors.

### DataFrames

Unlike in R, DataFrames are not part of the base language, nor the standard library. 

[DataFrames.jl](https://dataframes.juliadata.org/stable/) provides the basic infrastructure around DataFrames, as well as its own [mini language](https://bkamins.github.io/julialang/2020/12/24/minilanguage.html) for doing the split-apply-combine approach that underlies R's `dplyr` and much of the tidyverse.  The DataFrames.jl documentation is the place to for looking at how to e.g. read in a [CSV or Arrow file as a DataFrame](https://dataframes.juliadata.org/stable/man/importing_and_exporting/). Note that DataFrames.jl by default depends on [CategoricalArrays.jl](https://categoricalarrays.juliadata.org/stable/) to handle the equivalent of `factor` in the R world, but there is an alternative package for `factor`-like array type in Julia, [PooledArrays.jl](https://github.com/JuliaData/PooledArrays.jl/). PooledArrays are simpler, but more limited than CategoricalArrays and we (Phillip and Doug) sometimes use them in our examples and simulations.

DataFrame.jl's mini language can be a bit daunting, if you're used to manipulations in the style of base R or the tidyverse. For that, there are several options; recently, we'e had particularly nice experiences with [DataFrameMacros.jl](https://github.com/jkrumbiegel/DataFrameMacros.jl) and [Chain.jl](https://github.com/jkrumbiegel/Chain.jl) for a convenient syntax to connect or "pipe" together successive operations. It's your choice whether and which of these add-ons you want to use! Phillip tends to write his code using raw DataFrames.jl, but Doug really enjoys DataFrameMacros.jl. 
"""

# ╔═╡ 9143e623-5457-48b5-9304-ce2a757b18b5
md"""
## Regression

Unlike in R, neither formula processing nor basic regression are part of the base language or the standard library. 

The formula syntax and basic contrast-coding schemes in Julia is provided by [StatsModels.jl](https://juliastats.org/StatsModels.jl/v0.6/). By default, MixedModels.jl re-exports the `@formula` macro and most commonly used contrast schemes from StatsModels.jl, so you often don't have to worry about loading StatsModels.jl directly. The same is true for [GLM.jl](https://juliastats.org/GLM.jl/dev/manual/), which provides basic linear and generalized linear models, such as ordinary least squares (OLS) regression and logistic regression, i.e. the classical, non mixed regression models.

The basic functionality looks quite similar to R, e.g.

```julia
julia> lm(@formula(y ~ 1 + x), data)
julia> glm(@formula(y ~ 1 + x), data, Binomial(), LogitLink())
```

but the more general modelling API (also used by MixedModels.jl) is also supported:

```julia
julia> fit(LinearModel, @formula(y ~ 1 + x), mydata)
julia> fit(GeneralizedLinearModel, @formula(y ~ 1 + x), data, Binomial(), LogitLink())
```

(You can also specify your model matrices directly and skip the formula interface, but we don't recommend this as it's easy to mess up in really subtle but very probelmatic ways.)


### `@formula`, macros and domain-specific languages

As a sidebar: why is `@formula` a macro and not a normal function? Well, that's because formulas are essentially their own domain-specific language (a variant of [Wilkinson-Roger notation](https://www.jstor.org/stable/2346786)) and macros are used for manipulating the language itself -- or in this case, handling an entirely new, embedded language! This is also why macros are used by packages like [Turing.jl](https://turing.ml/) and [Soss.jl](https://cscherrer.github.io/Soss.jl/stable/) that define a language for Bayesian probabilistic programming like [PyMC3](https://docs.pymc.io/) or [Stan](https://mc-stan.org/).

### Extensions to the formula syntax

There are several ongoing efforts to extend the formula syntax to include some of the "extras" available in R, e.g. [RegressionFormulae.jl](https://github.com/kleinschmidt/RegressionFormulae.jl) to use the caret (`^`) notation to limit interactions to a certain order (`(a+b+c)^2` generates `a + b + c + a&b + a&c + b&c`, but not `a&b&c`). 
Note also that Julia uses `&` to express interactions, not `:` like in R.

### Standardizing Predictors

Although function calls such as `log` can be used within Julia formulae, they must act on a rowwise basis, i.e. on observations. Transformations such as z-scoring or centering (often done with `scale` in R) require knowledge of the entire column. [StandardizedPredictors.jl](https://beacon-biosignals.github.io/StandardizedPredictors.jl/stable/) provides functions for centering, scaling, and z-scoring within the formula. These are treated as pseudo-contrasts and computed on demand, meaning that `predict` and `effects` (see next) computations will handle these transformations on new data (e.g. centering new data *around the mean computed during fitting the original data*) correctly and automatically.

### Effects

John Fox's `effects` package in R (and the related `ggeffects` package for plotting these using `ggplot2`) provides a nice way to visualize a model's overall view of the data. This functionality is provided by [`Effects.jl`](https://beacon-biosignals.github.io/Effects.jl/stable/) and works out-of-the-box with most regression model packages in Julia (including MixedModels.jl). Support for formulae with embedded functions (such as `log`) is not yet complete, but we're working on it!

### Estimated Marginal / Least Square Means

To our knowledge, There is currently nothing like the R package `emmeans` package in Julia. However, it is often better to use sensible, hypothesis-driven contrast coding than to compute all pairwise comparisons after the fact. 😃
"""

# ╔═╡ 32c08c8a-68ec-4bcb-bd4b-3fd76ffca0f0
md"""
## Hypothesis Testing

Classical statistical tests such as the t-test can be found in the package [HypothesisTests.jl](https://github.com/JuliaStats/HypothesisTests.jl/).
"""

# ╔═╡ 4df36c00-be80-4fa5-83ca-c9648352fc42
md"""
## Plotting ecosystem

Throughtout this course, we have used the Makie ecosystem for plotting, but there are several alternatives in Julia.

### Makie

The [Makie ecosystem](https://makie.juliaplots.org/stable/) is a relatively new take on graphics that aims to be both powerful and easy to use. Makie.jl itself only provides abstract definitions for many components (and is used in e.g. MixedModelsMakie.jl to define plot types for MixedModels.jl). The actual plotting and rendering is handled by a backend package such as CairoMakie.jl (good for Pluto notebooks or rending static 2D images) and GLMakie.jl (good for dynamic, interactive visuals and 3D images). AlgebraOfGraphics.jl builds a grammar of graphics upon the Makie framework. It's a great way to get good plots very quickly, but extensive customization is still best achieved by using Makie directly.

### Plots.jl

[Plots.jl](https://docs.juliaplots.org/latest/) is the original plotting package in Julia, but we often find it difficult to work with compared to some of the other alternatives. [StatsPlots.jl](https://github.com/JuliaPlots/StatsPlots.jl) builds on this, adding common statistical plots, while [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl) renders plots as Unicode characters directly in the REPL.

[PGFPlotsX.jl](https://kristofferc.github.io/PGFPlotsX.jl/stable/) is a very new package that writes directly to PGF (the format used by LaTeX's tikz framework) and can stand alone or be used as a rendering backend for the Plots.jl ecosystem.

### Gadfly

[Gadfly.jl](https://gadflyjl.org/stable/) was the original attempt to create a plotting system in Julia based on the grammar of graphics (the "gg" in `ggplot2`). Development has largely stalled, but some functionality still exceeds AlgebraOfGraphics.jl, which has taken up the grammar of graphics mantle. Notably, the MixedModels.jl documentation still uses Gadfly as of this writing (early September 2021).

### Others

There are many [other graphics packages available in Julia](https://juliapackages.com/c/graphics), often wrapping well-established frameworks such as [VegaLite](https://www.queryverse.org/VegaLite.jl/stable/).
"""

# ╔═╡ 919a3d3c-1375-46ab-8507-ae884bb52cfb
md"""
## Connecting to Other Languages

Using Julia doesn't mean you have to leave all the packages you knew in other languages behind. In Julia, it's often possible to even easily and quickly invoke code from other languages *from within Julia*. 


[RCall.jl](https://juliainterop.github.io/RCall.jl/stable/gettingstarted/) provides a very convenient interface for interacting with R. [JellyMe4.jl](https://github.com/palday/JellyMe4.jl/) add support for moving MixedModels.jl and `lme4` models back and forth between the languages (which means that you can use `emmeans`, `sjtools`, `DHARMa`, `car`, etc. to examine MixedModels.jl models!). [RData.jl](https://github.com/JuliaData/RData.jl) provides support for reading `.rds` and `.rda` files from Julia, while [RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl) provides convenient access to many of the standard datasets provided by R and various R packages.

[PyCall.jl](https://github.com/JuliaPy/PyCall.jl/) provides a very convenient way for interacting with Python code and packages. [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) builds upon this foundation to provide support for Python's `matplotlib`. Similarly, [PyMNE.jl](
https://github.com/beacon-biosignals/PyMNE.jl) and [PyFOOOF.jl](https://github.com/beacon-biosignals/pyfooof.jl) provide some additional functionality to make interacting with MNE-Python and FOOOF from within Julia even easier than with vanilla PyCall. 

For MATLAB users, there is also [MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl)


[Cxx.jl](https://juliainterop.github.io/Cxx.jl/stable/) provides interoperability with C++. It also provides a C++ REPL mode, making it possible to treating C++ much more like a dynamic language than the traditional compiler toolchain would allow.

Support for calling C and Fortran is [part of the Julia standard library](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/).
"""

# ╔═╡ 35c0e59d-23ce-4b60-b80c-e619a50f990d
md"""

## Notebooks and Webpages

We haved [Pluto.jl](https://github.com/fonsp/Pluto.jl/) throughout this course because it provides a nice, easy way to interact with Julia, including automatic package management. [PlutoUI.jl](https://github.com/fonsp/PlutoUI.jl) provides some nice tools for interacting with Pluto notebooks and making them more dynamic (including the `TableOfContents()` in this one!). We also like that Pluto notebooks are essentially normal Julia code stored in plain text under the hood, which means that they play quite nice with version control systems such as Git. Sometimes, however, the reactive nature of Pluto (update any one cell and the changes propogate to all impacted cells) is not desireable.

An alternative notebook interface is provide by [Jupyter](https://jupyter.org/), formerly IPython. Indeed, the "Ju" actually stands for Julia. Jupyter support is provided by the package [IJulia.jl](https://julialang.github.io/IJulia.jl/stable/).

The [Weave.jl](https://weavejl.mpastell.com/stable/) package provides "Julia markdown", which is similar to knitr/RMarkdown or the the Sweave formats in R. Notably, Weave also provides support for converting between `jmd` files and Jupyter notebooks.

Similarly, [Literate.jl](https://fredrikekre.github.io/Literate.jl/v2/) is a simple package for literate programming (i.e. programming where documentation and code are "woven" together) and can generate Markdown, plain code and Jupyter notebook output. It is designed for simplicity and speed and is less extensive than Weave.jl and Documenter.jl.

[Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/) is the standard tool for building webpages from Julia documentation

[Books.jl](https://rikhuijzer.github.io/Books.jl/) is a package designed to offer somewhat similar functionality to the `bookdown` package in R.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╟─27130015-4f45-4ebd-9094-07ec00261e5f
# ╟─cd73a70f-4ce0-4ef8-87be-d3b319066602
# ╟─9143e623-5457-48b5-9304-ce2a757b18b5
# ╟─32c08c8a-68ec-4bcb-bd4b-3fd76ffca0f0
# ╟─4df36c00-be80-4fa5-83ca-c9648352fc42
# ╟─919a3d3c-1375-46ab-8507-ae884bb52cfb
# ╟─35c0e59d-23ce-4b60-b80c-e619a50f990d
# ╠═539bb62b-e4e3-4d72-837c-302ac9335e2e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
