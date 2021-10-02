### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 880ea5ac-a851-446e-8da9-d5f9f161a932
begin
	using AlgebraOfGraphics
	using AlgebraOfGraphics: linear
	using Arrow
	using CairoMakie
	using CategoricalArrays
	using Chain
	using DataFrames
	using DataFrameMacros
	using MixedModels
	using MixedModelsMakie
	using MixedModelsMakie: simplelinreg
	using PlutoUI         # user interface tools for Pluto notebooks
	using Random
	using Statistics
	using StatsBase
	CairoMakie.activate!(type="svg")
	TableOfContents()
	Base.show(io::IO, ::MIME"text/html", x::CategoricalArrays.CategoricalValue) = print(io, get(x)) 
end

# ╔═╡ eebef932-1d63-41e5-998f-9748379c43af
md"""
# Mixed Models Tutorial: Basics with Emotikon Project

This script uses a subset of data reported in Fühner, Golle, Granacher, & Kliegl (2021). Age and sex effects in physical fitness components of 108,295 third graders including 515 primary schools and 9 cohorts. [Scientific Reports 11:17566](https://rdcu.be/cwSeR)
To circumvent delays associated with model fitting we work with models that are less complex than those in the reference publication. All the data to reproduce the models in the publication are used here, too; the script requires only a few changes to specify the more complex models in the article. 

The script is structured in four main sections: 

1. **Setup** with reading and examing the data, plotting the main results, and specifying the contrasts for the fixed factor `Test`
2. a demonstration of **model complexification** to determine a parsimonious random-effect structure appropriate for and supported by the data, including also a quite elaborate demonstration of **principle component analyses (PCAs)** of levels (scores) and effects,
3. specification of **nested fixed effects or interactions** in the levels of another, superordinate factors,
4. a **Glossary of MixedModels.jl commands** to inspect the information generated for a fitted model object. 

## 1. Setup
### 1.0 Packages and functions
"""

# ╔═╡ ee85abd9-0172-44d3-b03a-c6a780d33c72
md"""
### 1.1 Readme

Number of scores: 525126 in './data/fggk21.arrow'

 1. Cohort: 9 levels; 2011-2019
 2. School: 515 levels 
 3. Child: 108295 levels; all children are between 8.0 and 8.99 years old
 4. Sex: "Girls" (n=55,086), "Boys" (n= 53,209)
 5. age: testdate - middle of month of birthdate 
 6. Test: 5 levels
     + Endurance (`Run`):  6 minute endurance run [m]; to nearest 9m in 9x18m field
     + Coordination (`Star_r`): star coordination run [m/s]; 9x9m field, 4 x diagonal = 50.912 m
     + Speed(`S20_r`): 20-meters sprint [m/s]
     + Muscle power low (`SLJ`): standing long jump [cm] 
     + Muscle power up (`BPT`): 1-kg medicine ball push test [m] 
 7. score - see units

 ### 1.2 Preprocessing

 #### 1.2.1 Read data
"""

# ╔═╡ 39443fb0-64ec-4a87-b072-bc8ad6fa9cf4
begin
	df = DataFrame(Arrow.Table("./data/fggk21.arrow"))
	df.Sex = categorical(df.Sex)
	levels!(df.Sex, ["Boys", "Girls"])
	df.Test = categorical(df.Test)
    levels!(df.Test, ["Run", "Star_r", "S20_r", "SLJ", "BPT"])
	recode!(df.Test, "Run" => "Endurance", "Star_r" => "Coordination",
		               "S20_r" => "Speed", "SLJ" => "PowerLOW", "BPT" => "PowerUP")
	describe(df)
end

# ╔═╡ 51b77ca3-374d-4701-b936-b4a40834cf70
md"""
#### 1.2.2 Transformations
We center `age` at 8.5 years and compute z-scores for each `Test`. With these variables the data frame `df` contains all variables used for the final model in the original publication.
"""

# ╔═╡ 7440c439-9d4c-494f-9ac6-18c9fb2fe144
begin
	transform!(df, :age, :age => (x -> x .- 8.5) => :a1); # centered age (linear)
	select!(groupby(df,  :Test), :, :score => zscore => :zScore); # z-score
	dfx = df[:, Not(:score)]   # drop score
end

# ╔═╡ 1e6fe7f6-9ce4-4b13-953b-bf7dc532e53a
md"""
#### 1.2.3 Extract a stratified subsample 

For the prupose of the tutorial, we extract a random sample of 1000 boys and 1000 girls. `Child`, `School`, and `Cohort` are grouping variables. Traditionally, they are called random factors because the units (levels) of the factor are assumed to be a random sample from the population of their units (levels).  

Cohort has only nine "groups" and could have been included as a set of polynomical fixed-effect contrasts rather than a random factor. This choice warrants a short excursion: The secular trends are very different for different tests and require the inclusion of interaction terms with `Test` contrasts (see Figure 4 in [Fühner et al. (2021)](https://rdcu.be/cwSeR). The authors opted to absorb these effects in cohort-related variance components for the `Test` contrasts and plan to address the details of secular changes in a separate analysis.

For complex designs, when they are in the theoretical focus of an article, factors and covariates should be specified as part of the fixed effects. If they are not in the theoretical focus, but serve as statistical control variables, they could be put in the RES - if supported by the data.
"""

# ╔═╡ 29d0a540-29ce-43de-9dfe-75e535ff94eb
md""" **Stratified sampling:**  We generate a `Child` table with information about children. `MersenneTwister(42)` specifies **42** as the seed for the random number generator to ensure reproducibility of the stratification. For a different pattern of results choose, for example, **84**. We randomly sample 1000 boys and 1000 girls from this table; they are stored in `samp`. Then, we extract the corresponding subset of these children's test scores from `df` and store them `dat`.  
"""

# ╔═╡ b3f32059-0d7b-4b66-8559-7c8a9f4d14d3

begin
	Child = unique(select(df, :Cohort, :School, :Child, :Sex, :age))
	sample = let rng = MersenneTwister(42)
	combine(groupby(Child, :Sex), x -> x[rand(rng, 1:nrow(x), 1000), :])
	end
	insamp(x) = x ∈ sample.Child
	dat = @subset(df, insamp(:Child))
end


# ╔═╡ 1f71f182-d840-43fd-98e5-721909e67b60
md"Due to missing scores for some tests we have about 2% less than 10,000 observtions." 

# ╔═╡ eea9c588-f5f8-4905-8774-7031162f9be0
md"""
### 1.3  No evidence for `age x Sex x Test` interaction
The main results are captured in the figure constructed in this section. We build it both for the full data and the stratified subset. 

**Important technical note: We make use of the special feature of Pluto notebooks that cells can be computed _out of order_. The scatter of means displayed in Figures 1 and 2 is computed in the cells of section 1.3.2.** 

#### 1.3.1  Figure(s) of interaction
The core results of the article are reported in Figure 2 of [Scientific Reports 11:17566](https://rdcu.be/cwSeR). In summary:

+ Main effects of `age` and `Sex`: There are developmental gains in the ninth year of life; boys outperform girls. There is no main effect of `Test` because of z-scoring.
+ Interactions of `Test` and `age`: Tests differ in how much children improve during the year (i.e., the magnitude of developmental gain), that is slopes depend on `Test`. 
+ Interactions of `Test` and `Sex`: The sex difference is test dependent, that is the difference between the slopes depends on `Test`.  
+ The most distinctive result is the absence of evidence for an `age x Sex x Test` interaction, that is the slopes for boys and girls are statistically parallel for each of the five tests.
"""

# ╔═╡ a41bc417-3ca7-4f14-be88-5a91d236e88f
md"""
_Figure 1._ Performance differences for **full set of data** between 8.0 and 9.2 years by sex in the five physical fitness tests presented as z-transformed data computed separately for each test. `Endurance` = cardiorespiratory endurance (i.e., 6-min-run test), `Coordination` = star-run test, `Speed` = 20-m linear sprint test, 
`PowerLOW` = power of lower limbs (i.e., standing long jump test), `PowerUP` = 
power of upper limbs (i.e., ball push test), SD = standard deviation. Points 
are binned observed child means; lines are simple regression fits to the observations.
"""

# ╔═╡ 2b1cd9cc-2122-4f2c-b945-7cbd0cd7ebc1
md"""
What do the results look like for the stratified subsample? Here the parallelism is much less clear. In the final LMM we test whether the two regression lines in each of the five panels are statistically parallel for this subset of data. That is, we test the interaction of `Sex` and `age` as nested within the levels of `Test`. Most people want to know the signficance of these five Sex x age interactions. 

The theoretical focus of the article, however, is on comparisons between tests displayed next to each other. We ask whether the degree of parallelism is statistically the same for `Endurance` and `Coordination` (H1), `Coordination` and `Speed` (H2), `Speed` and `PowerLOW` (H3), and `PowerLow` and `PowerUP` (H4). Hypotheses H1 to H4 require `Sequential Difference` contrasts c1 to c4 for `Test`; they are tested as fixed effects for``H1 x age x Sex`, `H2 x age x Sex`, `H3 x age x Sex`, and `H4 x age x Sex`. 
"""

# ╔═╡ 56093005-7728-44e7-b121-39cdefeb33f3
md"""
_Figure 2._ Performance differences for **subset of data** between 8.0 and 9.2 years by sex in the five physical fitness tests presented as z-transformed data computed separately for each test.  `Endurance` = cardiorespiratory endurance (i.e., 6-min-run test), `Coordination` = star-run test, `Speed` = 20-m linear sprint test, 
`PowerLOW` = power of lower limbs (i.e., standing long jump test), `PowerUP` = 
apower of upper limbs (i.e., ball push test), SD = standard deviation. Points 
are binned observed child means; lines are simple regression fits to the observations.
"""

# ╔═╡ 58d561e2-77c0-43c2-990d-008286de4e5c
md"""#### 1.3.2 Compute means
The means scattered in Figures 1 and 2 are based on age bins of boys' and girls'  test performances. The means are stored in data frames `df2` for the full and `dat2` for the subset.
"""

# ╔═╡ f80c6440-6bf0-4656-97f5-15b547d724da
begin
df2 = combine(
	groupby(
		select(df, :, :age => ByRow(x -> round(x; digits=1)) => :age),
		[:Sex, :Test, :age],
	),
	:zScore => mean => :zScore,
	:zScore => length => :n,
)
end

# ╔═╡ a450b88b-9ac4-4d44-9560-9e6d2e0dd8fa
begin
	design1 = mapping(:age, :zScore; color = :Sex, col = :Test)
	lines1 = design1 * linear()
	means1 = design1 * visual(Scatter, markersize=5)
	data(df2) * means1 + data(df) * lines1 |> draw
end

# ╔═╡ e9280e08-ace4-48cd-ad4a-55501f315d6a
begin
dat2 = 
	combine(
	groupby(
		select(dat, :, :age => ByRow(x -> round(x; digits=1)) => :age),
		[:Sex, :Test, :age],
	),
	:zScore => mean => :zScore,
	:zScore => length => :n,
)
end

# ╔═╡ 462bd92a-056a-4e5d-8fa6-215784e0fb9d
begin
	design2 = mapping(:age, :zScore; color = :Sex, col = :Test)
	lines2 = design2 * linear()
	means2 = design2 * visual(Scatter, markersize=5)
	data(dat2) * means2 + data(dat) * lines2 |> draw
end

# ╔═╡ 2574cf1c-4a9a-462b-a2f2-d599a8b42ec1
md"""
#### 1.3.2 Regression on `age` by `Sex` for each `Test`
Another set of relevant statistics are the slopes for the regression of performance on age for boys and girls in each of the five tests. The lines in Figures 1 and 2, however, are computed directly from the raw data with the `linear()` command. 
"""

# ╔═╡ 499c1b55-17ac-439b-a7a8-5311f15f6ff7
combine(groupby(df,[:Sex, :Test]), [:age, :zScore] => simplelinreg => :coef)

# ╔═╡ 385dad32-8179-4c7c-985b-f2a2bc7c3b2f
combine(groupby(dat,[:Sex, :Test]), [:age, :zScore] => simplelinreg => :coef)

# ╔═╡ c6c6f056-b8b9-4190-ac14-b900bafa04df
md"""
### 1.4 _SeqDiffCoding_ of `Test`

_SeqDiffCoding_ was used in the publication. This specification tests pairwise 
differences between the five neighboring levels of `Test`, that is: 

+ H1: `Star_r` - `Run` (2-1)
+ H2: `S20_r` - `Star_r` (3-2) 
+ H3: `SLJ` - `S20_r` (4-3)
+ H4: `BPT` - `SLJ` (5-4)

The levels were sorted such that these contrasts map onto four  _a priori_ hypotheses; in other words, they are _theoretically_ motivated pairwise comparisons. 
The motivation also encompasses theoretically motivated interactions with `Sex`. 
The order of levels can also be explicitly specified during contrast construction. 
This is very useful if levels are in a different order in the dataframe.

Note that random factors `Child`, `School`, and `Cohort` are declared as `Grouping` variables. Technically, this specification is required for variables with a very large number of levels (e.g., 100K+ children). We recommend the explicit specification for all random factors as a general coding style. 

The first command recodes names indicating the physical fitness components used in the above figures and tables back to the shorter actual test names. This reduces  clutter in LMM outputs. 
"""

# ╔═╡ c5326753-a03b-4739-a82e-90ffa7c1ebdb
begin
levels!(dat.Sex, ["Girls", "Boys"])
recode!(dat.Test, "Endurance" => "Run", "Coordination" => "Star_r",
		"Speed" => "S20_r", "PowerLOW" => "SLJ", "PowerUP" => "BMT")
	
contr = merge(
       Dict(nm => SeqDiffCoding() for nm in (:Test, :Sex)),
       Dict(nm => Grouping() for nm in (:Child, :School, :Cohort)),
   );
end

# ╔═╡ f65477ea-5227-4c8f-b2af-c11cb46a673c
md"""
The statistical disadvantage of _SeqDiffCoding_ is that the contrasts are not orthogonal, 
that is the contrasts are correlated. This is obvious from the fact that levels 2, 3, 
and 4 are all used in two contrasts. One consequence of this is that correlation parameters
estimated between neighboring contrasts (e.g., 2-1 and 3-2) are difficult to interpret. 
Usually, they will be negative because assuming some practical limitations on the overall 
range (e.g., between levels 1 and 3), a small "2-1" effect "correlates" negatively with a 
larger "3-2" effect for mathematical reasons. 

Obviously, the tradeoff between theoretical motivation and statistical purity is something 
that must be considered carefully when planning the analysis. 

Various options for contrast coding are the topic of the *MixedModelsTutorial\_contrasts\_emotikon.jl* and *MixedModelsTutorial\_contrasts\_kwdyz.jl*
notbebooks.
"""

# ╔═╡ f7d6782e-dcd3-423c-a7fe-c125d8e4f810
md"""
## 2. Model complexification

We fit and compare three LMMs with the same fixed-effect structure but increasing complexity of the random-effect structure for `School`. We ignore the other two random factors `Child` and `Cohort` to avoid undue delays when fitting the models.

1. LMM `m_ovi`: allowing only varying intercepts ("Grand Means");
2. LMM `m_zcp`: adding variance components (VCs) for the four `Test` contrasts, `Sex`, and `age` to LMM `m_ovi`, yielding the zero-correlation parameters LMM;
3. LMM `m_cpx`: adding correlation parameters (CPs) to LMM `m_zcp`; yielding a complex LMM. 

In a final part illustrate how to check whether the complex model is supported by the data, rather than leading to a singular fit and, if supported by the data, whether there is an increase in goodness of fit associated with the model complexification. 

### 2.1 LMM `m_ovi`

In its random-effect structure (RES) we  only vary intercepts (i.e., Grand Means) for `School` (LMM `m_ovi`), that is we allow that the schools differ in the average fitness of its children, average over the five tests.

It is well known that such a simple RES is likely to be anti-conservative with respect to fixed-effect test statistics. 
"""

# ╔═╡ 9a885065-59db-47a0-aa0a-8574f136a834
begin
	f_ovi = @formula zScore ~ 1 + Test * Sex * a1 + (1 | School);
	m_ovi = fit(MixedModel, f_ovi, dat, contrasts=contr)
end

# ╔═╡ d6a4f68a-bf45-4c66-8c3d-242f37874d87
md"""Is the model singular (overparameterized, degenerate)? In other words: Is the model not supported by the data?"""

# ╔═╡ f797c72c-12da-4f7c-9ad1-2b75f4cf3902
issingular(m_ovi)

# ╔═╡ d6594f70-f74f-4b8d-bef4-23f71acb469c
md"""Models varying only in intercepts are almost always supported by the data"""

# ╔═╡ 8a302aae-565e-4a5c-8285-881dd36d873d
md"""
### 2.2 LMM `m_zcp`

In this LMM we allow that schools differ not only in `GM`, but also in the size of the four contrasts defined for `Test`, in the difference between boys and girls (`Sex`) and the developmental gain children achieve within the third grade (`age`). 

We assume that there is covariance associated with these CPs beyond residual noise, that is we assume that there is no detectable evidence in the data that the CPs are different from zero.
"""

# ╔═╡ c5c3172f-9801-49c8-a915-6c78c7469ae6
begin
	f_zcp = @formula zScore ~ 1 + Test * Sex * a1 +
                     zerocorr(1 + Test + Sex + a1 | School);
	m_zcp = fit(MixedModel, f_zcp, dat, contrasts=contr)
end

# ╔═╡ 5a02bce6-cbd4-4bfa-b47e-f59abaea7003
md"""Depending on sampling, this model estimating variance components for `School`  may or may not be supported by the data."""


# ╔═╡ 55130a9b-a9b8-4c8a-923c-3df6e6759c6b
issingular(m_zcp)

# ╔═╡ 20f1f363-5d4e-45cf-8ed2-dbb97549bc22
md"""
### 2.3 LMM `m_cpx`

In the complex LMM investigated in this sequence we give up the assumption of zero-correlation between VCs. 

"""

# ╔═╡ 2db49116-5045-460b-bb9a-c4544333482e
begin
	f_cpx =  @formula zScore ~ 1 + Test * Sex * a1 +
	                          (1 + Test + Sex + a1| School)
	m_cpx = fit(MixedModel, f_cpx, dat, contrasts=contr)
end

# ╔═╡ 4b0bffc1-8b0f-4594-97f6-5449959bd716
md"""We also need to see the VCs and CPs of the random-effect structure (RES)."""

# ╔═╡ efa892f9-aff9-43e1-922d-d09f9062f172
VarCorr(m_cpx)

# ╔═╡ 9947810d-8d61-4d12-bf3c-37c8451f9779
issingular(m_cpx)

# ╔═╡ 565d97ca-99f3-44d5-bca6-49696003a46f
md"""
The complex model is may or may not be supported by the data.
"""

# ╔═╡ f6f8832b-5f62-4762-a8b0-6727e93b21a0
md"""
### 2.4 Model comparisons

The checks of model singularity indicate that the three models are supported by the data. Does model complexification also increase the goodness of fit or are we only fitting noise?

#### 2.4.1 LRT and goodness-of-fit statistics

As the thee models are strictly hierarchically nested, we compare them with a likelihood-ratio tests (LRT) and AIC and BIC goodness-of-fit statistics derived from them..
"""


# ╔═╡ f6b1dd53-718d-4455-80f3-9fbf14e03f54
MixedModels.likelihoodratiotest(m_ovi, m_zcp, m_cpx)

# ╔═╡ 2703966a-b129-49de-81d8-6bc8c5048dbe
begin
	mods = [m_ovi, m_zcp, m_cpx];
	gof_summary = DataFrame(dof=dof.(mods), deviance=deviance.(mods),
              AIC = aic.(mods), AICc = aicc.(mods), BIC = bic.(mods))
end

# ╔═╡ df57033d-3148-4a0a-9055-d4c8ede2a0d1
md"""
These statistics will depend on sampling. In general, smaller deviance, AIC, and BIC indicate an improvement in goodness of fit. Usually, χ² should be larger than the associated degrees of freedom; for AIC and BIC the decrease should amount to more than 5, according to some literature. Severity of meeting these criteria increases from deviance to AIC to BIC. Therefore, it is not always the case that the criteria are unanimous in their verdict. Basicly, the more confirmatory the analysis, the more one may go with deviance and AIC; for exploratory analyses the BIC is probably a better guide. There are grey zones here. 

#### 2.4.2 Comparing fixed effects of `m_ovi`, `m_zcp`, and `m_cpx`

We check whether enriching the RES changed the significance of fixed effects in the final model.  
"""


# ╔═╡ cd6623f0-2961-4871-995a-0cc2d75dd320
begin
	m_ovi_fe = DataFrame(coeftable(m_ovi));
	m_zcp_fe = DataFrame(coeftable(m_zcp)); 
	m_cpx_fe = DataFrame(coeftable(m_cpx)); 
    m_all = hcat(m_ovi_fe[:, [1, 2, 4]],
		        leftjoin(m_zcp_fe[:, [1, 2, 4]], m_cpx_fe[:, [1, 2, 4]],
			on = :Name, makeunique=true), makeunique=true);
	rename!(m_all, 
    Dict( "Coef." => "b_ovi", "Coef._2" => "b_zcp", "Coef._1" => "b_cpx",         
              "z" => "z_ovi",     "z_2" => "z_zcp",     "z_1" => "z_cpx"))
	m_all2 = round.( m_all[:, [:b_ovi, :b_zcp, :b_cpx, 
				               :z_ovi, :z_zcp, :z_cpx]], digits=2)
	m_all3 = hcat(m_all.Name,  m_all2)
end 

# ╔═╡ 71233622-14f3-45a8-a78b-4fc6ef0198f3
md"""
The three models usually do **not** differ in fixed-effect estimates. For main effects of `age` and `Sex`,  z-values decrease strongly with the complexity of the model (i.e., standard errors are larger). For other coefficients, the changes are not very large and not consistent.

In general, dropping significant variance components and/or correlation parameters may lead to anti-conservative estimates of fixed effects (e.g., [Schielzeth & Forstmeier, 2009](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2657178/)). Basically, some of the variance allocated to `age` and `Sex` in LMM `m_ovi` could also be due to differences between schools. This ambiguity increased the uncertainty of the respective fixed effects in the other two LMMs.
"""

# ╔═╡ 0aa76f57-10de-4e84-ae26-858250fb50a7
md"""### 2.5 Fitting an overparameterized LMM
The complex LMM was not overparameterized with respect to `School`, because there are over 400 schools in the data.  When the number of units (levels) of a grouping factor is small relative to the number of parameters we are trying to estimate, we often end up with an overparameterized / degenerate random-effect structure. 

As an illustration, we fit a full CP matrix for the `Cohort`. As there are only nine cohorts in the data, we may be asking too much to estimate 5*6/2 = 15 VC/CP parameters. 
"""

# ╔═╡ 09d2c2bb-9ce1-4288-a8b8-a0f67c6ce65a
begin
	f_cpxCohort =  @formula zScore ~ 1 + Test * a1 * Sex +
	                                (1 + Test | Cohort)
	m_cpxCohort = fit(MixedModel, f_cpxCohort, dat, contrasts=contr)
end

# ╔═╡ 227a7734-161c-4689-9ff4-d12a5d3ac362
VarCorr(m_cpxCohort)

# ╔═╡ 57263322-5eb6-4a34-8167-aec88ebf1970
issingular(m_cpxCohort)

# ╔═╡ ca1f28f8-57e8-4e85-aa1e-b184b8dee8f0
md"""The model is overparameterized with several CPs estimated between |.98| and |1.00|. How about the **zero-correlation parameter** (zcp) version of this LMM?"""

# ╔═╡ d5efd2c3-4890-4067-b6de-fe2aa4198cdb
begin
	f_zcpCohort =  @formula zScore ~ 1 + Test * a1 * Sex +
	                        zerocorr(1 + Test | Cohort)
	m_zcpCohort = fit(MixedModel, f_zcpCohort, dat, contrasts=contr)
end

# ╔═╡ 6654f9af-0235-40db-9ebb-05f8264a4f61
issingular(m_zcpCohort)

# ╔═╡ 43693b8b-0365-4177-8655-c4c00881d185
md"""This `zcpLMM`  is also singular. Three of the five VCs are estimated as zero. This raises the possibility that LMM `m_oviCohort`  might fit as well as LMM `m_zcpCohort`.
"""

# ╔═╡ a8267035-17ed-4f28-a5d5-d7729917d7dc
begin
	f_oviCohort =  @formula zScore ~ 1 + Test * a1 * Sex + (1 | Cohort)
	m_oviCohort = fit(MixedModel, f_oviCohort, dat, contrasts=contr)
end

# ╔═╡ b2e4c600-a4a2-419e-a14b-3ed50dc18724
issingular(m_oviCohort)

# ╔═╡ 4f34f0c8-ec27-4278-944e-1225f853a371
md"""This solves the problem with singularity, but does LMM `m_zcpCohort` fit noise relative to the LMM `m_oviCohort`?"""

# ╔═╡ a824d97b-e235-4411-a7cf-af123bfe6c4f
MixedModels.likelihoodratiotest(m_oviCohort, m_zcpCohort)

# ╔═╡ f992cf2c-b0c0-4a96-a51d-289bb6fbe317
begin
	gof_summary2 = DataFrame(dof=dof.([m_oviCohort, m_zcpCohort, m_cpxCohort]),
		deviance=deviance.([m_oviCohort, m_zcpCohort, m_cpxCohort]),
                AIC = aic.([m_oviCohort, m_zcpCohort, m_cpxCohort]), 
		      AICc = aicc.([m_oviCohort, m_zcpCohort, m_cpxCohort]), 
		        BIC = bic.([m_oviCohort, m_zcpCohort, m_cpxCohort]))
end

# ╔═╡ 47fd8482-81b5-4dd9-ac0a-67861a32e7ed
md"""Indeed, adding VCs is fitting noise. Again, the goodness of fit statistics unanimously favor the selection of the LMM `m_oviCohort`.

Not shown here, but the `Cohort`-related VCs for the `Test` contrasts could be estimated reliably for the **full** data. Thus, the small number of cohorts does not necessarily prevent the determination of reliable differences between tests across cohorts. What if we include VCs and CPs related to random factors `Child` and `School`?"""

# ╔═╡ e3cf1ee6-b64d-4356-af23-1dd5c7a0fec6
md"""
#### 2.6 Fitting the published LMM `m1` to the reduced data

**WARNING: The following LMMs `m1`, `m2`, etc. take a bit longer (e.g., close to 6 minutes in the Pluto notebook, close to 3 minutes in the REPL on a MacBook Pro).** 

LMM `m1` reported in Fühner et al. (2021) included random factors for `School`,
`Child`, and `Cohort`. The RES for `School` was specified like in LMM `m_cpx`. The
RES for `Child` included VCs and CPs for `Test`, but not for linear developmental
gain in the ninth year of life `a1` or `Sex`; they are between-`Child` effects. 

The RES for `Cohort` included only VCs, no CPs for `Test`. The _parsimony_ was due
to the small number of nine levels for this grouping factor.

Here we fit this LMM `m1` for the reduced data. For a different subset of similar size on MacBook Pro [13 | 15 | 16] this took [303 | 250 | 244 ] s; for LMM `m1a` (i.e., dropping 1 school-relate VC for `Sex`), times are  [212 | 165 | 160] s. The corresponding `lme4` times for LMM `m1` are [397  | 348 | 195]. 

Finally, times for fitting the full set of data --not in this script--, for LMM `m1`are [60 | 62 | 85] minutes (!); for LMM `m1a` the times were [46 | 48 | 34] minutes. It was  not possible to fit the full set of data with `lme4`; after about 13 to 18 minutes the  program stopped with:  `Error in eval_f(x, ...) : Downdated VtV is not positive definite.`
"""

# ╔═╡ 0dd0d060-81b4-4cc0-b306-dda7589adbd2
begin
	f1 =  @formula zScore ~ 1 + Test * a1 * Sex +
	                       (1 + Test + a1 + Sex | School) + (1 + Test | Child) +
	                        zerocorr(1 + Test | Cohort) 
	m1 = fit(MixedModel, f1, dat, contrasts=contr)
end

# ╔═╡ 5f7a0626-92af-4ec7-aa86-e3559efab511
VarCorr(m1)

# ╔═╡ b8728c1a-b20e-4e88-ba91-927b6272c6d5
issingular(m1)

# ╔═╡ 242d5f20-b42a-4043-9fe1-5b272cf60194
md"""Depending on the random number for stratified samplign, LMM `m1` may or may not be supported by the data. 

We also fit an alternative parameterization, estimating VCs and CPs for `Test` scores rather than `Test` effects by replacing the `1 + ...` in the RE terms with `0 + ...`.
"""

# ╔═╡ 67e74190-ea65-42a3-a668-6262012b0492
begin
	f2 =  @formula zScore ~ 1 + Test * a1 * Sex +
	                       (0 + Test + a1 + Sex | School) + (0 + Test | Child) +
	                        zerocorr(0 + Test | Cohort) 
	m2 = fit(MixedModel, f2, dat, contrasts=contr)
end

# ╔═╡ b9fab30a-fbcb-40f2-822f-52e8292cf77e
issingular(m2)

# ╔═╡ b4a7ae70-8e34-436b-84eb-0174b9ce7ded
md"""
Depending on the random number generator seed, the model may or may not be supported in the alternative parameterization of scores. The fixed-effects profile is not affected (see 2.8 below). 
"""

# ╔═╡ 25268edb-e253-4246-bc45-a88288e77daf
md"""**RK: The order of RE terms is critical. In formula `f2` the `zerocorr()` term must be placed last as shown. If it is placed first, School-related and Child-related CPs are estimated/reported (?) as zero. This was not the case for formula `m1`. Thus, it appears to be related to the `0`-intercepts in School and Child terms. Need a reprex.**
"""

# ╔═╡ f188d9af-c47d-4809-ad6e-d54f9a094067
VarCorr(m2)

# ╔═╡ a70fe575-51ce-45df-a0c0-12205a5efd88
md"""
### 2.7 Principle Component Analysis of Random Effect Structure (rePCA)

The `ìssingular()` command is sort of a shortcut for a quick inspection of the principle components (PCs) of the variance-covariance matrix of the RES. With the `MixedModels.PCA()` command, we also obtain information about the amount of cumulative variance accounted for as we add PCs. 

The output also provides PC loadings which may facilitate interpretation of the CP matrices (if estimated). This topic will be picked uo in a separate vignette. See also [Fühner et al. (2021)](https://rdcu.be/cwSeR) for an application.

#### 2.7.1 Effects in RES

For every random factor, `MixedModels.PCA()` extracts as many PCs as there are VCs. Therefore, the cumulation of variance across PCs within a random factor will always add up to 100% -- at the latest with the last VC, but, in the case of overparameterized LMMs, the ceiling will be reached earlier. The final PCs are usually quite small. 

PCs are extracted in the order of the amount of unique variance they account for. The first PC accounts for the largest and the final PC for the least  amount of variance. The number the PCs with percent variance above a certain threshold indicates the number of weighted composites needed and reflects the dimensionality of the orthogonal space within which (almost) all the variance can be accounted for. The weights for forming composite scores are the listed loadings. For ease of interpretation it is often useful to change the sign of some composite scores. 

The PCA for LMM `m1` shows that each of the five PCs for `Child` accounts for a non-zero percent of unique variance. 

For `School`  fewer than seven PCs have unique variance. The exact number depends on sampling. The overparameterization of `School` might be resolved when the CPs for `Sex` are dropped from the LMM. 

`Cohort` was estimated with CPs forced to zero. Therefore, the VCs were forced to be orthogonal; they already represent the PCA solution. However, depending on sampling, not all PCs may be identified for this random factor either.

Importantly, again depending on sampling, a non-singular fit does not imply that unique variance is associated with all PCs (i.e., not for last PC for  `School`). Embrace uncertainty!
"""

# ╔═╡ 3cf33a07-62b9-4068-9eea-1c04effe5716
MixedModels.PCA(m1)

# ╔═╡ 09c7f02c-bd47-4de4-b586-491aa0e3d584
md"""#### 2.7.2 Scores in RES

Now lets looks at the PCA results for the alternative parameterization of LMM `m2`.  It is important to note that the reparameterization to base estimates of VCs and CPs on scores rather than effects applies only to the `Test` factor (i.e., the first factor in the formula); VCs for `Sex` and `age` refer to the associated effects. 

Depending on  sampling, the difference between LMM `m1` and LMM `m2` may show that overparameterization according to PCs may depend on the specification chosen for the other the random-effect structure. 

**Note: For the _complete_ data, all PCs had unique variance associated with them.**
"""

# ╔═╡ 6c350be3-d451-4ee1-a13c-c7b5498f8726
MixedModels.PCA(m2)

# ╔═╡ 920edea4-f2a0-4c50-af52-326832310802
md"""### 2.8 Summary of results for stratified subset of data

Returning to the theoretical focus of the article, the significant main effects of `age` and `Sex`, the interactions between `age` and c1 and c4 contrasts and the interactions between `Sex` and three test contrasts (c1, c2, c4) are replicated. Obviously, the subset of data is much noisier than the full set.
"""

# ╔═╡ 4572b20e-19e2-45e8-ba45-d4eb55cd3f4a
md"""
## 3. `Age x Sex` nested in levels of `Test`

In this final LMM, we test _post-hoc_  five `age x Sex` interactions by nesting the interaction in the levels of `Test`. As this LMM `m2_nested` is a reparameterization of LMM `m2`. 
"""

# ╔═╡ 801ba580-31c5-4d04-8557-afa86586048f

begin
f2_nested = @formula zScore ~ 1 + Test + Test & (a1*Sex) +
							   (0+Test+a1+Sex | School) + (0+Test | Child) + 
	   							zerocorr(0+Test | Cohort)
m2_nested = fit(MixedModel, f2_nested, dat, contrasts=contr)
end

# ╔═╡ cfc83241-1b91-4baf-8210-a9d48489894d
md"""
The results show that none of the interactions in the panels of Figure 2 is significant.  The size and direction of interaction effects correspond with what is shown in Figure 2.
"""

# ╔═╡ de972c0c-ee10-4cf0-a5a5-c99951abee7e
md"#### CONSTRUCTION SITE: More model comparisons"

# ╔═╡ ccc3e244-cdc6-42d1-b49a-46b2aca9ac01
begin
	gof_summary3 = DataFrame(dof=dof.([m1, m2, m2_nested]),
		deviance=deviance.([m1, m2, m2_nested]),
                AIC = aic.([m1, m2, m2_nested]), 
		      AICc = aicc.([m1, m2, m2_nested]), 
		        BIC = bic.([m1, m2, m2_nested]))
end

# ╔═╡ 2825d2e2-7171-45fd-9cc8-8a7fb014e3e2
n, p, q, k = size(m1)  # nobs, fe params, VCs+CPs, re terms

# ╔═╡ 5c32009c-cd09-4b58-89c1-3688907012f5
md"""
In prinicple, the models should yield the save deviance. When models are not supported by the data, that is for singular models, there may be small differences between deviances for these reparameterizations. During optimization such models search for the absolute minimum in a very shallow surface and  may end up in a local minimum instead.
"""

# ╔═╡ 5f16f834-167c-41f5-91c3-2830940b179d
md"""#### Geometric degrees of freedom / 

From MixedModels documentation: "The sum of the leverage values is the rank of the model matrix and `n - sum(leverage(m))` is the degrees of freedom for residuals. The sum of the leverage values is also the trace of the so-called "hat" matrix`H`." 

New term: _geometric degrees of freedom_. 
"""

# ╔═╡ 3e9fbde8-b1b4-4691-9346-68f3d416113a
m1_geomdf = sum(leverage(m1))  # geom_dof

# ╔═╡ 4f05562d-4c9d-4133-b529-347d5bf9519a
sum(leverage(m2))          

# ╔═╡ fb64580e-8fd0-4c2e-b9f8-9b1a4915bfc4
sum(leverage(m2_nested))

# ╔═╡ 5b6ffa3a-6ce8-4db4-8b5d-d017c11760dd
n - m1_geomdf

# ╔═╡ f6b03391-e248-4e8d-b2fd-dbd7dab206f9
m1.feterm.rank

# ╔═╡ 1a26069b-6df7-4e96-8d3d-24e520946c4e
dof(m1)

# ╔═╡ 46e4c42d-fc89-4d12-b700-6c3f087e64ca
md"""
## 4. Glossary of _MixedModels.jl_ commands

Here we introduce most of the commands available in the _MixedModels.jl_ 
package that allow the immediated inspection and analysis of results returned in a fitted _linear_ mixed-effect model. 

Postprocessing related to conditional modes will be dealt with in a different tutorial.

### 4.1 Overall summary statistics

```
+ julia> m1.optsum         # MixedModels.OptSummary:  gets all info 
+ julia> loglikelihood(m1) # StatsBase.loglikelihood: return loglikelihood
							 of the model
+ julia> deviance(m1)      # StatsBase.deviance: negative twice the log-likelihood
							 relative to saturated model
+ julia> objective(m1)     # MixedModels.objective: saturated model not clear:
							 negative twice the log-likelihood
+ julia> nobs(m1)          # n of observations; they are not independent
+ julia> dof(m1)           # n of degrees of freedom is number of model parameters
+ julia> aic(m1)           # objective(m1) + 2*dof(m1)
+ julia> bic(m1)           # objective(m1) + dof(m1)*log(nobs(m1))
```
"""


# ╔═╡ 0caf1da1-4d06-49b8-b1b4-d24a860c68d8
m1.optsum            # MixedModels.OptSummary:  gets all info

# ╔═╡ 117132af-b6bb-4b11-a92d-3fc34aff3a63
loglikelihood(m1) # StatsBase.loglikelihood: return loglikelihood of the model

# ╔═╡ e707d474-c553-4142-8b40-d6d944cbc58a
deviance(m1)      # StatsBase.deviance: negative twice the log-likelihood relative to saturated mode`

# ╔═╡ ba7c1050-6b29-4dc2-a97b-552bdc259ac0
objective(m1)    # MixedModels.objective: saturated model not clear: negative twice the log-likelihood

# ╔═╡ 80abc5b9-9d9d-4a67-a542-1be2dc1eca0f
nobs(m1) # n of observations; they are not independent

# ╔═╡ ccac476f-f046-44b8-b65b-06dd975097b4
n_, p_, q_, k_ = size(m1)

# ╔═╡ d413615b-1ea8-46b2-837c-52a8f147b456
dof(m1)  # n of degrees of freedom is number of model parameters

# ╔═╡ 0f2b176f-64c6-4b90-98fe-5ddfb255dc93
geom_df = sum(leverage(m1)) # trace of hat / rank of model matrix / geom dof

# ╔═╡ e2057518-e508-42c0-ba92-c9bb0afeec20
resid_df = nobs(m1) - geom_df  # eff. residual degrees of freedom

# ╔═╡ 2a58411c-c975-4477-8e76-0b9b59000e75
aic(m1)  # objective(m1) + 2*dof(m1)

# ╔═╡ 1caa69ec-f1db-4de5-a30a-3d65fd45f5bd
bic(m1)  # objective(m1) + dof(m1)*log(nobs(m1))

# ╔═╡ f300cd09-9abe-4dcf-91e2-dbba7e87b8c1
md"""
### 4.2 Fixed-effect statistics
```
+ julia> coeftable(m1)     # StatsBase.coeftable: fixed-effects statiscs; 
						     default level=0.95
+ julia> Arrow.write("./data/m_cpx_fe.arrow", DataFrame(coeftable(m1)));
+ julia> coef(m1)          # StatsBase.coef - parts of the table
+ julia> fixef(m1)         # MixedModels.fixef: not the same as coef() 
                             for rank-deficient case
+ julia> m1.beta           # alternative extractor
+ julia> fixefnames(m1)    # works also for coefnames(m1)
+ julia> vcov(m1)          # StatsBase.vcov: var-cov matrix of fixed-effects coef.
+ julia> stderror(m1)      # StatsBase.stderror: SE for fixed-effects coefficients
+ julia> propertynames(m1) # names of available extractors
```
"""


# ╔═╡ 0cc6ca34-7699-4a85-a174-6fdf1a985613
coeftable(m1) # StatsBase.coeftable: fixed-effects statiscs; default level=0.95

# ╔═╡ 222380a0-8b36-4920-a70d-0fa480005748
#Arrow.write("./data/m_cpx_fe.arrow", DataFrame(coeftable(m1)));

# ╔═╡ 0c998d91-d4cf-4c71-8f4b-38ac21f8169e
coef(m1)              # StatsBase.coef; parts of the table

# ╔═╡ b7c715ed-4d11-4256-8ce6-0793388d9dfc
fixef(m1)    # MixedModels.fixef: not the same as coef() for rank-deficient case

# ╔═╡ d9e6af5c-a302-481b-a8a9-2f48d01884a5
m1.β                  # alternative extractor

# ╔═╡ 56e4cd41-1cf3-416c-856a-9cc939fe51cb
fixefnames(m1)        # works also for coefnames(m1)

# ╔═╡ 75999511-1600-4ce5-8bb7-6dc841bbbd50
vcov(m1)   # StatsBase.vcov: var-cov matrix of fixed-effects coefficients

# ╔═╡ 815c070a-8cc1-40f3-91bb-7c38f904399b
vcov(m1; corr=true) # StatsBase.vcov: correlation matrix of fixed-effects coefficients

# ╔═╡ 4eef686a-6943-45a8-8309-e1b264af34b5
stderror(m1)       # StatsBase.stderror: SE for fixed-effects coefficients

# ╔═╡ 89a1cfec-2b87-4bf9-95e3-7c7312069af8
propertynames(m1)  # names of available extractors

# ╔═╡ 1ee16dfe-97d6-4958-b9c4-cf2812691057
md"""
### 5.3 Covariance parameter estimates
These commands inform us about the model parameters associated with the RES.
```
+ julia> issingular(m1)        # Test singularity for param. vector m1.theta
+ julia> VarCorr(m1)           # MixedModels.VarCorr: est. of RES
+ julia> propertynames(m1)
+ julia> m1.σ                  # residual; or: m1.sigma
+ julia> m1.σs                 # VCs; m1.sigmas
+ julia> m1.θ                  # Parameter vector for RES (w/o residual); m1.theta
+ julia> MixedModels.sdest(m1) #  prsqrt(MixedModels.varest(m1))
+ julia> BlockDescription(m1)  #  Description of blocks of A and L in an LMM
```
"""

# ╔═╡ 7302b067-270f-4d6a-a0c6-fabd46cc72b0
issingular(m1) # Test if model is singular for paramter vector m1.theta (default)

# ╔═╡ a6286166-7eed-4f8a-9df6-4f389052e679
issingular(m2)

# ╔═╡ acd9a3eb-6eae-4e09-8932-e44f91e725b4
VarCorr(m1) # MixedModels.VarCorr: estimates of random-effect structure (RES)

# ╔═╡ b99fd996-a043-41e1-b142-7a91bb6bd9eb
VarCorr(m2)

# ╔═╡ 1c9de228-93a4-4d1a-82b5-48fe22cfe0f1
m1.σs      # VCs; m1.sigmas

# ╔═╡ 363dcab6-f9f7-42e7-9de7-6a9616cb060f
m1.θ       # Parameter vector for RES (w/o residual); m1.theta

# ╔═╡ f0538b16-e32e-4393-81d2-1d0ce271d619
BlockDescription(m1) #  Description of blocks of A and L in a LinearMixedModel

# ╔═╡ 8715ec4f-53bd-4b7f-8442-74223d3f4ecc
m2.θ 

# ╔═╡ d2b4b01c-33fd-4f8c-a2b0-c5dd298a5367
BlockDescription(m2)

# ╔═╡ ae8aae41-58bd-49b6-a56e-d05cad10c22f
md"""
### 5.4 Model "predictions" 
These commands inform us about extracion of conditional modes/means and (co-)variances, that using the model parameters to improve the predictions for units (levels) of the grouping (random) factors. We need this information, e.g., for partial-effect response profiles (e.g., facet plot) or effect profiles (e.g., caterpillar plot), or visualizing the borrowing-strength effect for correlation parameters (e.g., shrinkage plots). We are using the fit of LMM `m2`.


```
+ julia> condVar(m2)
```

Some plotting functions are currently available from the `MixedModelsMakie` package or via custom functions.

```
+ julia> caterpillar!(m2, orderby=1)
+ julia> shrinkage!(m2)
```

"""

# ╔═╡ 3b24e3f0-4f93-42d5-ba93-eeb23a1e8bd3
md" #### 5.4.1 Conditional covariances"

# ╔═╡ da3c1ba5-55cf-41f1-8cd6-8b2784532476
begin
cv_m1 = condVar(m1)
cv_m2 = condVar(m2)
end

# ╔═╡ b2e83fe2-b60d-415d-8577-c0958bfe8d0e
md"""
They are hard to look at. Let's take pictures.
#### 5.4.2 Caterpillar plots
"""

# ╔═╡ 80feb95f-4eb1-4363-8223-1d42f9c637a9
begin	# Cohort
	cm_m1_chrt = ranefinfo(m1)[:Cohort];
	caterpillar!(Figure(; resolution=(800,400)), cm_m1_chrt; orderby=1)
end

# ╔═╡ e07e768b-a1ba-438a-8082-de818e798565
md"#### 5.4.3 Shrinkage plots"

# ╔═╡ 91db2051-b222-41c4-96c5-a2fa0c977bfa
md" These are just teasers. We will pick this up in a separate tutorial. Enjoy! "

# ╔═╡ b044860a-c571-4dcb-bc3c-d5b07fe58695
shrinkageplot!(Figure(; resolution=(800,800)), m1, :Cohort)

# ╔═╡ c4138607-9c46-471e-ad58-c603a7cbdc58
shrinkageplot!(Figure(; resolution=(800,800)), m2, :Cohort)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrameMacros = "75880514-38bc-4a95-a458-c2aea5a3a702"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MixedModels = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
MixedModelsMakie = "b12ae82c-6730-437f-aff9-d2c38332a376"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.5.4"
Arrow = "~1.6.2"
CairoMakie = "~0.6.5"
CategoricalArrays = "~0.10.1"
Chain = "~0.4.8"
DataFrameMacros = "~0.1.1"
DataFrames = "~1.2.2"
MixedModels = "~4.3.0"
MixedModelsMakie = "~0.3.9"
PlutoUI = "~0.7.10"
StatsBase = "~0.33.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[AlgebraOfGraphics]]
deps = ["Colors", "Dates", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "KernelDensity", "Loess", "Makie", "PlotUtils", "PooledArrays", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "3746ac120f8b163dbd3bda0f36b6dec20c4f795a"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.5.4"

[[Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "b8d49c34c3da35f220e7295659cd0bab8e739fed"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.33"

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

[[Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "61adeb0823084487000600ef8b1c00cc2474cd47"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.0"

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

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "StaticArrays"]
git-tree-sha1 = "8664989955daccc90002629aa80193e44893bb45"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.6.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "fbc5c413a005abdeeb50ad0e54d85d000a1ca667"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.1"

[[Chain]]
git-tree-sha1 = "cac464e71767e8a04ceee82a889ca56502795705"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.8"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bd4afa1fdeec0c8b89dad3c6e92bc6e3b0fec9ce"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.6.0"

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

[[ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "45efb332df2e86f2cb2e992239b6267d97c9e0b6"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.7"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "1a90210acd935f222ea19657f143004d2c2a1117"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.38.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrameMacros]]
deps = ["DataFrames"]
git-tree-sha1 = "5cea6ee03472aebeb7bb375a862cbe0dc50b139f"
uuid = "75880514-38bc-4a95-a458-c2aea5a3a702"
version = "0.1.1"

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

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f4efaa4b5157e0cdb8283ae0b5428bc9208436ed"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.16"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7f6ad1a7f4621b4ab8e554133dade99ebc6e7221"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.5"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "StaticArrays"]
git-tree-sha1 = "19d0f1e234c13bbfd75258e55c52aa1d876115f5"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.2"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[GeoInterface]]
deps = ["RecipesBase"]
git-tree-sha1 = "38a649e6a52d1bea9844b382343630ac754c931c"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "0.5.5"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "2c1cf4df419938ece72de17f368a021ee162762e"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Match", "Observables"]
git-tree-sha1 = "e2f606c87d09d5187bb6069dab8cee0af7c77bdb"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.6.1"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[HypertextLiteral]]
git-tree-sha1 = "72053798e1be56026b81d4e2682dbe58922e5ec9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.0"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "13c826abd23931d909e4c5538643d9691f62a617"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.8"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

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

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

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

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b5254a86cf65944c68ed938e575f5c81d5dfe4cb"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.3"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "7e49f989e7c7f50fe55bd92d45329c9cf3f2583d"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.15.2"

[[MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "7bcc8323fb37523a6a51ade2234eee27a11114c8"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.1.3"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Match]]
git-tree-sha1 = "5cf525d97caf86d29307150fcba763a64eaa9cbe"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.1.0"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "4819019da9f42746851ddc39d90222d1dbe953b7"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.10.3"

[[MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test"]
git-tree-sha1 = "70e733037bbf02d691e78f95171a1fa08cdc6332"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.2.1"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[MixedModels]]
deps = ["Arrow", "DataAPI", "Distributions", "GLM", "JSON3", "LazyArtifacts", "LinearAlgebra", "Markdown", "NLopt", "PooledArrays", "ProgressMeter", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StatsModels", "StructTypes", "Tables"]
git-tree-sha1 = "a21b216d060abda9ad2cbbc4ac94d085c937f8e0"
uuid = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
version = "4.3.0"

[[MixedModelsMakie]]
deps = ["DataFrames", "Distributions", "KernelDensity", "LinearAlgebra", "Makie", "MixedModels", "Printf", "SpecialFunctions", "StatsBase"]
git-tree-sha1 = "c5c3b33a28878e573b35fb98724e2158249d6679"
uuid = "b12ae82c-6730-437f-aff9-d2c38332a376"
version = "0.3.9"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "748f6e1e4de814b101911e64cc12d83a6af66782"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.2"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "f115030b9325ca09ef1619ba0617b2a64101ce84"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.4"

[[NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "2b597c46900f5f811bec31f0dcc88b45744a2a09"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.0+0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e14c485f6beee0c7a8dcf6128bf70b85f1fe201e"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.9"

[[Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9bc1871464b12ed19297fbc56c4fb4ba84988b0d"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.47.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "9d8c00ef7a8d110787ff6f170579846f776133a9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.4"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[PlutoUI]]
deps = ["Base64", "Dates", "HypertextLiteral", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "26b4d16873562469a0a1e6ae41d90dec9e51286d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.10"

[[PolygonOps]]
git-tree-sha1 = "c031d2332c9a8e1c90eca239385815dc271abb22"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.1"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

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

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "0529f4188bc8efee85a7e580aca1c7dff6b103f8"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

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

[[SIMD]]
git-tree-sha1 = "9ba33637b24341aba594a2783a502760aa0bff04"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.3.1"

[[ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "9cc2955f2a254b18be655a4ee70bc4031b2b189e"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

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

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

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
deps = ["ChainRulesCore", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ad42c30a6204c74d264692e633133dcea0e8b14e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.2"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "a8f30abc7c64a39d389680b74e749cf33f872a70"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.3"

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
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "1bc8cc83e458c8a5036ec7206a04d749b9729fe8"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.26"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "8445bf99a36d703a09c601f9a57e2f83000ef2ae"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

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
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "632a8d4dbbad6627a4d2d21b1c6ebcaeebb1e1ed"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.2"

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

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[isoband_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "a1ac99674715995a536bbce674b068ec1b7d893d"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.2+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─eebef932-1d63-41e5-998f-9748379c43af
# ╠═880ea5ac-a851-446e-8da9-d5f9f161a932
# ╟─ee85abd9-0172-44d3-b03a-c6a780d33c72
# ╠═39443fb0-64ec-4a87-b072-bc8ad6fa9cf4
# ╟─51b77ca3-374d-4701-b936-b4a40834cf70
# ╠═7440c439-9d4c-494f-9ac6-18c9fb2fe144
# ╟─1e6fe7f6-9ce4-4b13-953b-bf7dc532e53a
# ╟─29d0a540-29ce-43de-9dfe-75e535ff94eb
# ╠═b3f32059-0d7b-4b66-8559-7c8a9f4d14d3
# ╟─1f71f182-d840-43fd-98e5-721909e67b60
# ╟─eea9c588-f5f8-4905-8774-7031162f9be0
# ╠═a450b88b-9ac4-4d44-9560-9e6d2e0dd8fa
# ╟─a41bc417-3ca7-4f14-be88-5a91d236e88f
# ╟─2b1cd9cc-2122-4f2c-b945-7cbd0cd7ebc1
# ╠═462bd92a-056a-4e5d-8fa6-215784e0fb9d
# ╟─56093005-7728-44e7-b121-39cdefeb33f3
# ╟─58d561e2-77c0-43c2-990d-008286de4e5c
# ╠═f80c6440-6bf0-4656-97f5-15b547d724da
# ╠═e9280e08-ace4-48cd-ad4a-55501f315d6a
# ╟─2574cf1c-4a9a-462b-a2f2-d599a8b42ec1
# ╠═499c1b55-17ac-439b-a7a8-5311f15f6ff7
# ╠═385dad32-8179-4c7c-985b-f2a2bc7c3b2f
# ╟─c6c6f056-b8b9-4190-ac14-b900bafa04df
# ╠═c5326753-a03b-4739-a82e-90ffa7c1ebdb
# ╟─f65477ea-5227-4c8f-b2af-c11cb46a673c
# ╟─f7d6782e-dcd3-423c-a7fe-c125d8e4f810
# ╠═9a885065-59db-47a0-aa0a-8574f136a834
# ╟─d6a4f68a-bf45-4c66-8c3d-242f37874d87
# ╠═f797c72c-12da-4f7c-9ad1-2b75f4cf3902
# ╟─d6594f70-f74f-4b8d-bef4-23f71acb469c
# ╟─8a302aae-565e-4a5c-8285-881dd36d873d
# ╠═c5c3172f-9801-49c8-a915-6c78c7469ae6
# ╠═5a02bce6-cbd4-4bfa-b47e-f59abaea7003
# ╠═55130a9b-a9b8-4c8a-923c-3df6e6759c6b
# ╟─20f1f363-5d4e-45cf-8ed2-dbb97549bc22
# ╠═2db49116-5045-460b-bb9a-c4544333482e
# ╟─4b0bffc1-8b0f-4594-97f6-5449959bd716
# ╠═efa892f9-aff9-43e1-922d-d09f9062f172
# ╠═9947810d-8d61-4d12-bf3c-37c8451f9779
# ╟─565d97ca-99f3-44d5-bca6-49696003a46f
# ╟─f6f8832b-5f62-4762-a8b0-6727e93b21a0
# ╠═f6b1dd53-718d-4455-80f3-9fbf14e03f54
# ╠═2703966a-b129-49de-81d8-6bc8c5048dbe
# ╟─df57033d-3148-4a0a-9055-d4c8ede2a0d1
# ╟─cd6623f0-2961-4871-995a-0cc2d75dd320
# ╟─71233622-14f3-45a8-a78b-4fc6ef0198f3
# ╟─0aa76f57-10de-4e84-ae26-858250fb50a7
# ╠═09d2c2bb-9ce1-4288-a8b8-a0f67c6ce65a
# ╠═227a7734-161c-4689-9ff4-d12a5d3ac362
# ╠═57263322-5eb6-4a34-8167-aec88ebf1970
# ╟─ca1f28f8-57e8-4e85-aa1e-b184b8dee8f0
# ╠═d5efd2c3-4890-4067-b6de-fe2aa4198cdb
# ╟─6654f9af-0235-40db-9ebb-05f8264a4f61
# ╟─43693b8b-0365-4177-8655-c4c00881d185
# ╠═a8267035-17ed-4f28-a5d5-d7729917d7dc
# ╟─b2e4c600-a4a2-419e-a14b-3ed50dc18724
# ╟─4f34f0c8-ec27-4278-944e-1225f853a371
# ╠═a824d97b-e235-4411-a7cf-af123bfe6c4f
# ╠═f992cf2c-b0c0-4a96-a51d-289bb6fbe317
# ╟─47fd8482-81b5-4dd9-ac0a-67861a32e7ed
# ╟─e3cf1ee6-b64d-4356-af23-1dd5c7a0fec6
# ╠═0dd0d060-81b4-4cc0-b306-dda7589adbd2
# ╠═5f7a0626-92af-4ec7-aa86-e3559efab511
# ╠═b8728c1a-b20e-4e88-ba91-927b6272c6d5
# ╟─242d5f20-b42a-4043-9fe1-5b272cf60194
# ╠═67e74190-ea65-42a3-a668-6262012b0492
# ╠═b9fab30a-fbcb-40f2-822f-52e8292cf77e
# ╟─b4a7ae70-8e34-436b-84eb-0174b9ce7ded
# ╟─25268edb-e253-4246-bc45-a88288e77daf
# ╠═f188d9af-c47d-4809-ad6e-d54f9a094067
# ╟─a70fe575-51ce-45df-a0c0-12205a5efd88
# ╠═3cf33a07-62b9-4068-9eea-1c04effe5716
# ╟─09c7f02c-bd47-4de4-b586-491aa0e3d584
# ╠═6c350be3-d451-4ee1-a13c-c7b5498f8726
# ╟─920edea4-f2a0-4c50-af52-326832310802
# ╟─4572b20e-19e2-45e8-ba45-d4eb55cd3f4a
# ╠═801ba580-31c5-4d04-8557-afa86586048f
# ╟─cfc83241-1b91-4baf-8210-a9d48489894d
# ╟─de972c0c-ee10-4cf0-a5a5-c99951abee7e
# ╟─ccc3e244-cdc6-42d1-b49a-46b2aca9ac01
# ╠═2825d2e2-7171-45fd-9cc8-8a7fb014e3e2
# ╟─5c32009c-cd09-4b58-89c1-3688907012f5
# ╟─5f16f834-167c-41f5-91c3-2830940b179d
# ╠═3e9fbde8-b1b4-4691-9346-68f3d416113a
# ╠═4f05562d-4c9d-4133-b529-347d5bf9519a
# ╠═fb64580e-8fd0-4c2e-b9f8-9b1a4915bfc4
# ╠═5b6ffa3a-6ce8-4db4-8b5d-d017c11760dd
# ╠═f6b03391-e248-4e8d-b2fd-dbd7dab206f9
# ╠═1a26069b-6df7-4e96-8d3d-24e520946c4e
# ╟─46e4c42d-fc89-4d12-b700-6c3f087e64ca
# ╠═0caf1da1-4d06-49b8-b1b4-d24a860c68d8
# ╠═117132af-b6bb-4b11-a92d-3fc34aff3a63
# ╠═e707d474-c553-4142-8b40-d6d944cbc58a
# ╠═ba7c1050-6b29-4dc2-a97b-552bdc259ac0
# ╠═80abc5b9-9d9d-4a67-a542-1be2dc1eca0f
# ╠═ccac476f-f046-44b8-b65b-06dd975097b4
# ╠═d413615b-1ea8-46b2-837c-52a8f147b456
# ╠═0f2b176f-64c6-4b90-98fe-5ddfb255dc93
# ╠═e2057518-e508-42c0-ba92-c9bb0afeec20
# ╠═2a58411c-c975-4477-8e76-0b9b59000e75
# ╠═1caa69ec-f1db-4de5-a30a-3d65fd45f5bd
# ╟─f300cd09-9abe-4dcf-91e2-dbba7e87b8c1
# ╠═0cc6ca34-7699-4a85-a174-6fdf1a985613
# ╠═222380a0-8b36-4920-a70d-0fa480005748
# ╠═0c998d91-d4cf-4c71-8f4b-38ac21f8169e
# ╠═b7c715ed-4d11-4256-8ce6-0793388d9dfc
# ╠═d9e6af5c-a302-481b-a8a9-2f48d01884a5
# ╠═56e4cd41-1cf3-416c-856a-9cc939fe51cb
# ╠═75999511-1600-4ce5-8bb7-6dc841bbbd50
# ╠═815c070a-8cc1-40f3-91bb-7c38f904399b
# ╠═4eef686a-6943-45a8-8309-e1b264af34b5
# ╠═89a1cfec-2b87-4bf9-95e3-7c7312069af8
# ╟─1ee16dfe-97d6-4958-b9c4-cf2812691057
# ╠═7302b067-270f-4d6a-a0c6-fabd46cc72b0
# ╠═a6286166-7eed-4f8a-9df6-4f389052e679
# ╠═acd9a3eb-6eae-4e09-8932-e44f91e725b4
# ╠═b99fd996-a043-41e1-b142-7a91bb6bd9eb
# ╠═1c9de228-93a4-4d1a-82b5-48fe22cfe0f1
# ╠═363dcab6-f9f7-42e7-9de7-6a9616cb060f
# ╠═f0538b16-e32e-4393-81d2-1d0ce271d619
# ╠═8715ec4f-53bd-4b7f-8442-74223d3f4ecc
# ╠═d2b4b01c-33fd-4f8c-a2b0-c5dd298a5367
# ╟─ae8aae41-58bd-49b6-a56e-d05cad10c22f
# ╟─3b24e3f0-4f93-42d5-ba93-eeb23a1e8bd3
# ╠═da3c1ba5-55cf-41f1-8cd6-8b2784532476
# ╟─b2e83fe2-b60d-415d-8577-c0958bfe8d0e
# ╠═80feb95f-4eb1-4363-8223-1d42f9c637a9
# ╟─e07e768b-a1ba-438a-8082-de818e798565
# ╟─91db2051-b222-41c4-96c5-a2fa0c977bfa
# ╠═b044860a-c571-4dcb-bc3c-d5b07fe58695
# ╠═c4138607-9c46-471e-ad58-c603a7cbdc58
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
