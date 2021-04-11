### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ d17d5cf8-988a-11eb-03dc-23f1449f5563
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 4cd88e2b-c6f0-49cb-846a-eaae9f892e02
begin
	using Arrow, DataAPI, DataFrames, MixedModels
	using DataAPI: levels
end

# ╔═╡ 4aee40fa-7b43-48fe-8c95-e2a62253755a
using ImageIO, Images, ImageMagick

# ╔═╡ 6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
md"""
# Mixed Models Tutorial: Basics

Ths script uses data reported in Fühner, Golle, Granacher, & Kliegl (2021). 
Physical fitness in third grade of primary school: 
A mixed model analysis of 108,295 children and 515 schools.

To circumvent delays associated with model fitting we work with models that are less complex than those in the reference publication. All the data to reproduce the models in the publication are used here, too; the script requires only a few changes to specify the more complex models in the paper. 

The script is structured in three main sections: 

1. The **Setup** with reading and examing the data, plotting the main results, and specifying the contrasts for the fixed factor `Test`.
2. A demonstration of **Model complexification** to determine a random-effect structure appropriate for and supported by the data.
3. A **Glossary of MixedModels.jl commands** to inspect the information generated for a fitted model object. 

## 1. Setup

### 1.1 Readme for 'fggk21.rds'

1. Cohort: 9 levels; 2011-2019
2. District ("Landkreis"): 18 levels
3. BU: 2 levels; school is close ("B") or far ("U") from Berlin
4. School: 515 levels 
5. Child: 108295 levels; all children were between 8.0 and 8.99 years at legal keydate (30 September).
6. Sex:
7. age: test date - middle of month of birthdate 
8. Test: 5 levels
     + Endurance (`Run`):  6 minute endurance run [m]; to nearest 9m in 9x18m field
     + Coordination (`Star_r`): star coordination run [m/s]; 9x9m field, 4 x diagonal = 50.912 m
     + Speed(`S20_r`): 20-meters sprint [m/s]
     + Muscle power low (`SLJ`): standing long jump [cm] 
     + Muscle power up (`BPT`): 1-kg medicine ball push test [m] 
 9. score - see units
 10. zScore - this variable will be used as DV for comparison across tests

### 1.2 Read and examine data
"""

# ╔═╡ d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
dat = Arrow.Table("./data/fggk21.arrow");

# ╔═╡ 8e122db8-416c-4ee7-b3c2-6e829659a606
describe(DataFrame(dat))

# ╔═╡ 032da1d7-dae3-4146-bf79-e7678ea32b32
levels(dat.Test)

# ╔═╡ b8aff2cd-c3ab-44af-81cd-5c8cb037caa1
levels(dat.Sex)

# ╔═╡ 80d9dbe0-8232-43d4-ba17-827bc823ea58
md"""
### 1.3 Plot main results
"""

# ╔═╡ 19530903-70c3-492d-bdf3-87bdb90abbea
md"""
The main results of relevance for this tutorial are shown in this figure from Fühner et al. (2021, Figure 2). There are developmental gains within the ninth year of life for each of the five tests and the tests differ in the magnitude of these gains. Also boys outperform girls on each test and, again, the sex difference varies across test.
"""

# ╔═╡ 3a1f6150-d2ee-4a82-8909-7c50073507cc
begin
	fig1 = load("./figures/Figure_2.png");
	fig1
end


# ╔═╡ 25df1635-ab21-4bfb-bb17-8336b19f5f62
md"""
_Figure 1._ Performance differences between 8.0 und 9.0 years by sex in the five physical fitness tests presented as z-transformed data computed separately for each test. Endurance = cardiorespiratory endurance (i.e., 6 min run test), Coordination = star run test, Speed = 20-m linear sprint test, PowerLOW = power of lower limbs (i.e., standing long jump test), PowerUP = power of upper limbs (i.e., ball push test), SD = standard deviation. Points are binned observed child means; lines are simple regression fits to the observations; 95% confidence intervals for means ≈ 0.05 are not visible.
"""

# ╔═╡ 632f4d33-97f0-4861-b93c-7996c555c00e
md"""
### 1.4 _SeqDiffCoding_ of `Test`

_SeqDiffCoding_ was used in the publication. This specification tests pairwise differences between the five neighboring levels of `Test`, that is: 

+ H1: `Star_r` - `Run` (2-1)
+ H2: `S20_r` - `Star_r` (3-2) 
+ H3: `SLJ` - `S20_r` (4-3)
+ H4: `BPT` - `SLJ` (5-4)

The levels were sorted such that these contrasts map onto four  _a priori_ hypotheses; in other words, they are _theoretically_ motivated pairwise comparisons. The motivation also encompasses theoretically motivated interactions with `Sex`. The order of levels can also be explicitly specified during contrast construction. This is very useful if levels are in a different order in the dataframe.

The statistical disadvantage of _SeqDiffCoding_ is that the contrasts are not orthogonal, that is the contrasts are correlated. This is obvious from the fact that levels 2, 3, and 4 are all used in two contrasts. One consequence of this is that correlation parameters estimated between neighboring contrasts (e.g., 2-1 and 3-2) are difficult to interpret. Usually, they will be negative because assuming some practical limitations on the overall range (e.g., between levels 1 and 3), a small "2-1" effect "correlates" negatively with a larger "3-2" effect for mathematical reasons. 

Obviously, the tradeoff between theoretical motivation and statistical purity is something that must be considered carefully when planning the analysis. 

Various options for contrast coding are the topic of the _MixedModelsTutorial_contrasts.jl_ notbebook.
"""


# ╔═╡ 1fea85b8-58ae-481a-a6d8-6da198a81bd8
contr = merge(
		   Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		   Dict(:Sex => EffectsCoding(; levels=["female", "male"])),
	       Dict(:Test => SeqDiffCoding(; levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"]))
	
	   )

# ╔═╡ d813057a-ddce-4bd5-bf40-f15b71c6eeed
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

# ╔═╡ 36c0a967-6d5e-4771-8790-95224371d9ba
begin
	f_ovi = @formula zScore ~ 1 + Test * (Sex + (age - 8.5)) + (1 | School);
	m_ovi = fit(MixedModel, f_ovi, dat, contrasts=contr)
end

# ╔═╡ 80bcdcb9-0dc8-49e8-8640-7d7123ca4bb2
md"""
### 2.2 LMM `m_zcp`

In this LMM we allow that schools differ not only in `GM`, but also in the size of the four contrasts defined for `Test`, in the difference between boys and girls (`Sex`) and the developmental gain children achieve within the third grade (`age`). 

We assume that there is covariance associated with these CPs beyond residual noise, that is we assume that there is no detectable evidence in the data that the CPs are different from zero.

"""

# ╔═╡ 605b4000-a52e-4419-a908-3178dbb4c91e
begin
	f_zcp = @formula zScore ~ 1 + Test * (Sex + (age - 8.5)) +
                     zerocorr(1 + Test +  Sex + (age - 8.5) | School);
	m_zcp = fit(MixedModel, f_zcp, dat, contrasts=contr)
end

# ╔═╡ e5b27ff8-e0d3-4026-8cf8-c10f8aa1d54e
md"""Is the model singular (overparameterized, degenerate)? In other words: Is the model **not** supported by the data?"""

# ╔═╡ 471070af-8b39-41ba-b604-a88ae119a5d5
issingular(m_zcp)

# ╔═╡ 2cf4bc78-7ac0-47b4-8661-a0a7ea52b470
md"""
### 2.3 LMM `m_cpx`

In the most complex LMM investigated in this sequence we give up the assumption of zero-correlation between VCs. 

"""

# ╔═╡ 4499aa2f-725c-469f-b34c-d96a280fe20e
begin
	f_cpx =  @formula zScore ~ 1 + Test * (Sex + (age - 8.5)) +
	                          (1 + Test +  Sex + (age - 8.5) | School)
	m_cpx = fit(MixedModel, f_cpx, dat, contrasts=contr)
end

# ╔═╡ 84c845a0-ed62-480f-9970-b0e1c101732d
VarCorr(m_cpx)

# ╔═╡ 92931850-058b-4215-ac3a-c61eb7d2e4c9
md"""
The  CPs associated with `Sex` are very small. Indeed, they could be removed from the LMM. The VC for `Sex`, however, is needed. The specification of these hierarchical  model comparisons within LMM `m_cpx` is left as an online demonstration.

Is the model singular (overparameterized, degenerate)? In other words: Is the model **not** supported by the data?
"""

# ╔═╡ 8e32a0cb-143a-4bda-9a3c-7d21bb82612b
issingular(m_cpx)

# ╔═╡ efaa575a-d03e-431b-a745-8553e095e111
md"""
### 2.4 Model comparisons

The checks of model singularity indicate the all three models are supported by the data. Does model complexification also increase the goodness of fit or are we only fitting noise?

The models are strictly hierarchically nested. We compare the three LMM with a likelihood-ratio tests (LRT) and AIC and BIC goodness-of-fit statistics.

"""

# ╔═╡ d2e8cbd4-1c65-450e-83d9-caeba7b8f926
MixedModels.likelihoodratiotest(m_ovi, m_zcp, m_cpx)

# ╔═╡ 7cb282db-37dd-497f-9d73-edc563a105ef
mods = [m_ovi, m_zcp, m_cpx];

# ╔═╡ fb8f1bbf-7b7a-43b3-bedd-2977b533ffc4
gof_summary = DataFrame(dof=dof.(mods), deviance=deviance.(mods),
              AIC = aic.(mods), AICc = aicc.(mods), BIC = bic.(mods))

# ╔═╡ 81decdea-8d98-453b-8d2b-a5ded6d815fd
md"""
The likelihood-ratio test (LRT) and the most conservatice BIC statistic agree that with an increase with model complexity. Therefore, both the additions of VCs and the addition of CPs significantly improved the goodness of fit.

We check whether enriching the RES changed the significance of fixed effects in the final model.  

**<TO BE DONE**: Assemble FE statistics from the three LMMs for easy comparison**>**
"""

# ╔═╡ cdcdd7ad-466b-457c-b23f-f042d3ffa8ac
m_cpx

# ╔═╡ ce5d4ea2-e687-4bd0-8377-d3bf21f99721
md"""
## 3. Glossary of _MixedModels.jl_ commands

Here we introduce most of the commands available in the _MixedModels.jl_ 
pacakge that allow the immediated inspection and analysis of results returned in a fitted _linear_ mixed-effect model. 

Postprocessing related to conditional modes will be dealt with in a different tutorial.
"""

# ╔═╡ 61bab316-4cea-4f5e-b82e-120d98f6d7a8
md"""### 3.1 Overall summary statistics"""

# ╔═╡ b955207b-d422-4579-b32c-e4273710d7ce
# MixedModels.OptSummary: get all info with: m1.optsum)

loglikelihood(m_cpx)  # StatsBase.loglikelihood: return loglikelihood of the model

# ╔═╡ 98766877-fe20-4fac-a689-dd66d476cd40
deviance(m_cpx)   # StatsBase.deviance: negative twice the log-likelihood relative to saturated model

# ╔═╡ 11291f17-1647-4b0f-8aad-26a24c75dfd1
objective(m_cpx)  # MixedModels.objective: saturated model not clear: negative twice the log-likelihood

# ╔═╡ 36c392c0-548a-4cc6-8786-34e626a71cc3
# all from StatsBase
nobs(m_cpx) # n of observations; they are not independent

# ╔═╡ 7eb6ef49-ec67-4649-9d91-da4d90e82db7
dof(m_cpx)  # n of degrees of freedom is number of model parameters

# ╔═╡ 755ea5ac-bf09-4ead-9f00-0c1eb973554e
aic(m_cpx)  # objective(m1) + 2*dof(m1)

# ╔═╡ bd69fae6-2d00-42e7-80f4-021c3638c043
bic(m_cpx)  # objective(m1) + dof(m1)*log(nobs(m1))

# ╔═╡ c114fd0c-7031-4e20-ac6b-96ae59c58eca
md"""
### 3.2 Fixed-effect statistics
"""

# ╔═╡ 25fb4af8-4c9b-4df4-9946-d870ed65a75a
coeftable(m_cpx)     # StatsBase.coeftable: fixed-effects statiscs; default level=0.95

# ╔═╡ 90cecc7a-6fee-41f8-9b59-971d236f4130
Arrow.write("./data/m1_fe.arrow", DataFrame(coeftable(m_cpx)));

# ╔═╡ 4a2e0a60-1361-44db-a476-abcc3e2ced83
# ... or parts of the table
coef(m_cpx)              # StatsBase.coef

# ╔═╡ 20b9633d-1691-4c3b-9579-5355aac858d2
fixef(m_cpx)    # MixedModels.fixef: not the same as coef() for rank-deficient case

# ╔═╡ 274f3585-c78b-4900-a664-8ad588c3d59d
m_cpx.β                  # alternative extractor

# ╔═╡ 1b60d8f0-70bf-410b-9785-f30119ff6908
fixefnames(m_cpx)        # works also for coefnames(m1)

# ╔═╡ 2cc9f330-dcd8-40a1-8839-b81dbc99c08f
vcov(m_cpx)   # StatsBase.vcov: var-cov matrix of fixed-effects coefficients

# ╔═╡ ed5667e4-7691-451e-9c7b-4ddd367c4d48
vcov(m_cpx; corr=true) # StatsBase.vcov: correlation matrix of fixed-effects coefficients

# ╔═╡ 973da4fc-3d95-469b-868b-33d933c43957
stderror(m_cpx)          # StatsBase.stderror: SE for fixed-effects coefficients

# ╔═╡ da417df8-c674-4035-bb62-3c59ba18e349
propertynames(m_cpx)  # names of available extractors

# ╔═╡ a55dc7ed-2e34-4fdf-a51b-b4c170538a89
md"""
## 3.3 Covariance parameter estimates
These commands inform us about the model parameters associated with the RES.
"""

# ╔═╡ e48923d7-be76-45bd-98de-cbecd82f4640
BlockDescription(m_cpx) #  Description of blocks of A and L in a LinearMixedModel

# ╔═╡ 12ce94a6-4e04-40cd-bbad-2f5e8a766f77
VarCorr(m_cpx) # MixedModels.VarCorr: estimates of random-effect structure (RES)

# ╔═╡ 55e8ac8e-2c95-4e96-921f-4a3a79fa17f9
#propertynames(m1)
m_cpx.σ     # residual; m1.sigma, MixedModels.sdest(m1), sqrt(MixedModels.varest(m1))

# ╔═╡ 18daa693-e91f-4f72-a551-5c2e7d040de6
m_cpx.σs    # VCs; m1.sigmas

# ╔═╡ 9ee525bb-a497-4796-94f2-418f81e8c53b
m_cpx.θ     # Parameter vector for RES (w/o residual); m1.theta

# ╔═╡ 7f1928ee-91d8-41f9-8e1c-44c35fc33fd9
## check singularity
issingular(m_cpx) # Test if model is singular for paramter vector m1.theta (default)

# ╔═╡ Cell order:
# ╠═d17d5cf8-988a-11eb-03dc-23f1449f5563
# ╟─6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
# ╠═4cd88e2b-c6f0-49cb-846a-eaae9f892e02
# ╠═d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
# ╠═8e122db8-416c-4ee7-b3c2-6e829659a606
# ╠═032da1d7-dae3-4146-bf79-e7678ea32b32
# ╠═b8aff2cd-c3ab-44af-81cd-5c8cb037caa1
# ╟─80d9dbe0-8232-43d4-ba17-827bc823ea58
# ╠═4aee40fa-7b43-48fe-8c95-e2a62253755a
# ╟─19530903-70c3-492d-bdf3-87bdb90abbea
# ╠═3a1f6150-d2ee-4a82-8909-7c50073507cc
# ╟─25df1635-ab21-4bfb-bb17-8336b19f5f62
# ╟─632f4d33-97f0-4861-b93c-7996c555c00e
# ╠═1fea85b8-58ae-481a-a6d8-6da198a81bd8
# ╟─d813057a-ddce-4bd5-bf40-f15b71c6eeed
# ╠═36c0a967-6d5e-4771-8790-95224371d9ba
# ╟─80bcdcb9-0dc8-49e8-8640-7d7123ca4bb2
# ╠═605b4000-a52e-4419-a908-3178dbb4c91e
# ╟─e5b27ff8-e0d3-4026-8cf8-c10f8aa1d54e
# ╠═471070af-8b39-41ba-b604-a88ae119a5d5
# ╟─2cf4bc78-7ac0-47b4-8661-a0a7ea52b470
# ╠═4499aa2f-725c-469f-b34c-d96a280fe20e
# ╠═84c845a0-ed62-480f-9970-b0e1c101732d
# ╟─92931850-058b-4215-ac3a-c61eb7d2e4c9
# ╠═8e32a0cb-143a-4bda-9a3c-7d21bb82612b
# ╟─efaa575a-d03e-431b-a745-8553e095e111
# ╠═d2e8cbd4-1c65-450e-83d9-caeba7b8f926
# ╠═7cb282db-37dd-497f-9d73-edc563a105ef
# ╠═fb8f1bbf-7b7a-43b3-bedd-2977b533ffc4
# ╟─81decdea-8d98-453b-8d2b-a5ded6d815fd
# ╠═cdcdd7ad-466b-457c-b23f-f042d3ffa8ac
# ╟─ce5d4ea2-e687-4bd0-8377-d3bf21f99721
# ╟─61bab316-4cea-4f5e-b82e-120d98f6d7a8
# ╠═b955207b-d422-4579-b32c-e4273710d7ce
# ╠═98766877-fe20-4fac-a689-dd66d476cd40
# ╠═11291f17-1647-4b0f-8aad-26a24c75dfd1
# ╠═36c392c0-548a-4cc6-8786-34e626a71cc3
# ╠═7eb6ef49-ec67-4649-9d91-da4d90e82db7
# ╠═755ea5ac-bf09-4ead-9f00-0c1eb973554e
# ╠═bd69fae6-2d00-42e7-80f4-021c3638c043
# ╟─c114fd0c-7031-4e20-ac6b-96ae59c58eca
# ╠═25fb4af8-4c9b-4df4-9946-d870ed65a75a
# ╠═90cecc7a-6fee-41f8-9b59-971d236f4130
# ╠═4a2e0a60-1361-44db-a476-abcc3e2ced83
# ╠═20b9633d-1691-4c3b-9579-5355aac858d2
# ╠═274f3585-c78b-4900-a664-8ad588c3d59d
# ╠═1b60d8f0-70bf-410b-9785-f30119ff6908
# ╠═2cc9f330-dcd8-40a1-8839-b81dbc99c08f
# ╠═ed5667e4-7691-451e-9c7b-4ddd367c4d48
# ╠═973da4fc-3d95-469b-868b-33d933c43957
# ╠═da417df8-c674-4035-bb62-3c59ba18e349
# ╟─a55dc7ed-2e34-4fdf-a51b-b4c170538a89
# ╠═e48923d7-be76-45bd-98de-cbecd82f4640
# ╠═12ce94a6-4e04-40cd-bbad-2f5e8a766f77
# ╠═55e8ac8e-2c95-4e96-921f-4a3a79fa17f9
# ╠═18daa693-e91f-4f72-a551-5c2e7d040de6
# ╠═9ee525bb-a497-4796-94f2-418f81e8c53b
# ╠═7f1928ee-91d8-41f9-8e1c-44c35fc33fd9
