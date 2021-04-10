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
	using Arrow, DataAPI, DataFrames, MixedModels, StatsBase
	using DataAPI: levels
end

# ╔═╡ 6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
md"""
# Mixed Models Tutorial: Physical fitness in third grade

Ths script uses data reported in Fühner, Golle, Granacher, & Kliegl (2021). 
Physical fitness in third grade of primary school: 
A mixed model analysis of 108,295 children and 515 schools.

It contains if not at all so definitely most of the commands available in the MixedModels.jl 
pacakge for the analysis of _linear_ mixed-effect models.

The models analyzed here are using simpler random-effect structures than those in the 
publication to speed up LMM estimations; they are still quite complex and not or not easily 
handled by other software. 

## Readme for 'fggk21.rds'

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

## Read and transform

+ read and examine data
+ specify contrasts for the fixed-effects factors `Test` and `Sex`
+ set the "contrasts" for the random-effects grouping factors to `Grouping()`
"""

# ╔═╡ d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
dat = Arrow.Table("./data/fggk21.arrow");

# ╔═╡ 8e122db8-416c-4ee7-b3c2-6e829659a606
describe(DataFrame(dat))

# ╔═╡ 7dca92f9-dcc2-462c-b501-9ecabce74005
contr = merge(
       Dict(nm => HelmertCoding() for nm in (:Test, :Sex)),
       Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
   );

# ╔═╡ d813057a-ddce-4bd5-bf40-f15b71c6eeed
md"""
## Hierarchical tests of VC and CP for Child and School

Two sequences of tests to check that VCs and CPs increase goodness of fit for random facters `Child` and `School`.

### LMM - only varying intercepts
"""

# ╔═╡ e41bb345-bd0a-457d-8d57-1e27dde43a63
f_ovi = @formula zScore ~ 1 + Test * (age - 8.5) * Sex + 
        (1 | School) + (1 | Child) + (1 | Cohort);

# ╔═╡ baf13d52-1569-4077-af4e-40aa53d38cf5
m_ovi = fit(MixedModel, f_ovi, dat, contrasts=contr)

# ╔═╡ 153c3cb8-2488-4d70-9236-a5d4e47b6f1b
md"""
### Focus on Child

+ add VC to oviLMM
+ add CP to VC
+ compare with reference LMM
"""

# ╔═╡ 9a3aa079-5bba-49ef-af84-6fd3e82a1ee7
f_VC_C = @formula zScore ~ 1 + Test * (age - 8.5) * Sex + 
         (1 | School) + zerocorr(1 + Test | Child) + (1 | Cohort);

# ╔═╡ 08763df8-aea2-4de7-bbb0-b4a1247c541a
m_VC_C = fit(MixedModel, f_VC_C, dat, contrasts=contr)

# ╔═╡ a230a7d8-d4a7-4d5a-8e02-69d14ee157fc
f_CP_C = @formula zScore ~ 1 + Test * (age - 8.5) * Sex + 
         (1 | School) + (1 + Test | Child) + (1 | Cohort);

# ╔═╡ dc38aff4-71a1-44e8-8154-a00efa9aa2cb
m_CP_C = fit(MixedModel, f_CP_C, dat, contrasts=contr)

# ╔═╡ 60f522a1-27aa-48fb-af89-83cb2ce38cae
md"""
### Compare goodness of fit for Child sequence
"""

# ╔═╡ 0aac930e-4b2d-4b49-9590-1d21ca0b3217
MixedModels.likelihoodratiotest(m_ovi, m_VC_C, m_CP_C)

# ╔═╡ 7cb282db-37dd-497f-9d73-edc563a105ef
mods = [m_ovi, m_VC_C, m_CP_C];

# ╔═╡ fb8f1bbf-7b7a-43b3-bedd-2977b533ffc4
gof_summary = DataFrame(dof=dof.(mods), deviance=deviance.(mods),
              AIC = aic.(mods), AICc = aicc.(mods), BIC = bic.(mods))

# ╔═╡ 81decdea-8d98-453b-8d2b-a5ded6d815fd
md"""
Both VCs and CPs significantly improve the goodness of fit.

### Focus on School

+ add VC to oviLMM
+ add CP to VC
+ compare with reference LMM
"""

# ╔═╡ 71c49ff4-e207-40f3-ba6c-ff5aa248557c
f_VC_S = @formula zScore ~ 1 + Test * (age - 8.5) * Sex + 
         zerocorr(1 + Test | School) + (1 | Child) + (1 | Cohort);

# ╔═╡ 534cc633-a02c-4ede-9d4b-d7ad46e38fd3
m_VC_S = fit(MixedModel, f_VC_S, dat, contrasts=contr)

# ╔═╡ 804b8775-31ec-486f-88c6-0586d93c1666
f_CP_S = @formula zScore ~ 1 + Test * (age - 8.5) * Sex + 
         (1 + Test | School) + (1 | Child) + (1 | Cohort);

# ╔═╡ bc72326f-0ecf-44d1-ba4f-688ae7361662
m_CP_S = fit(MixedModel, f_CP_S, dat, contrasts=contr)

# ╔═╡ efaa575a-d03e-431b-a745-8553e095e111
md"### Compare goodness of fit for School sequence"

# ╔═╡ ee1cad9d-ebc8-4bb6-918a-9d38f0cd2232
MixedModels.likelihoodratiotest(m_ovi, m_VC_S, m_CP_S)

# ╔═╡ 10c85aa3-9243-4c62-9041-90e78d27d3a9
mods_S = [m_ovi, m_VC_S, m_CP_S];

# ╔═╡ 327f3fd1-0118-4952-8e0d-0779f69ad819
gof_summary_S = DataFrame(dof=dof.(mods_S), deviance=deviance.(mods_S),
              AIC = aic.(mods_S), AICc = aicc.(mods_S), BIC = bic.(mods_S))

# ╔═╡ 2a5a7755-baf2-479e-9a57-4de137b5ecf3
md"""
Both VCs and CPs significantly improve the goodness of fit.

## LMM
"""


# ╔═╡ 4c0a5cac-913d-490e-aeb9-505d34d89b5c
f1 = @formula zScore ~ 1 + Test * (age - 8.5) * Sex +
     (1+Test+(age - 8.5)+Sex | School) +  (1+Test | Child) + (1 | Cohort);

# ╔═╡ 8fbaf19e-869e-4cff-84a8-83023b572e87
m1 = fit(MixedModel, f1, dat, contrasts=contr)

# ╔═╡ ce5d4ea2-e687-4bd0-8377-d3bf21f99721
md"## Fit statistics"

# ╔═╡ b955207b-d422-4579-b32c-e4273710d7ce
# MixedModels.OptSummary: get all info with: m1.optsum)

loglikelihood(m1)  # StatsBase.loglikelihood: return loglikelihood of the model

# ╔═╡ 98766877-fe20-4fac-a689-dd66d476cd40
deviance(m1)   # StatsBase.deviance: negative twice the log-likelihood relative to saturated model

# ╔═╡ 11291f17-1647-4b0f-8aad-26a24c75dfd1
objective(m1)  # MixedModels.objective: saturated model not clear: negative twice the log-likelihood

# ╔═╡ 36c392c0-548a-4cc6-8786-34e626a71cc3
# all from StatsBase
nobs(m1) # n of observations; they are not independent

# ╔═╡ 7eb6ef49-ec67-4649-9d91-da4d90e82db7
dof(m1)  # n of degrees of freedom is number of model parameters

# ╔═╡ 755ea5ac-bf09-4ead-9f00-0c1eb973554e
aic(m1)  # objective(m1) + 2*dof(m1)

# ╔═╡ bd69fae6-2d00-42e7-80f4-021c3638c043
bic(m1)  # objective(m1) + dof(m1)*log(nobs(m1))

# ╔═╡ 3838c638-9214-4ddd-b834-85dd1b2b1ce5
md"## Fixed effects"

# ╔═╡ 25fb4af8-4c9b-4df4-9946-d870ed65a75a
coeftable(m1)     # StatsBase.coeftable: fixed-effects statiscs; default level=0.95

# ╔═╡ 90cecc7a-6fee-41f8-9b59-971d236f4130
Arrow.write("./data/m1_fe.arrow", DataFrame(coeftable(m1)));

# ╔═╡ 4a2e0a60-1361-44db-a476-abcc3e2ced83
# ... or parts of the table
coef(m1)              # StatsBase.coef

# ╔═╡ 20b9633d-1691-4c3b-9579-5355aac858d2
fixef(m1)    # MixedModels.fixef: not the same as coef() for rank-deficient case

# ╔═╡ 274f3585-c78b-4900-a664-8ad588c3d59d
m1.β                  # alternative extractor

# ╔═╡ 1b60d8f0-70bf-410b-9785-f30119ff6908
fixefnames(m1)        # works also for coefnames(m1)

# ╔═╡ 2cc9f330-dcd8-40a1-8839-b81dbc99c08f
vcov(m1)   # StatsBase.vcov: var-cov matrix of fixed-effects coefficients

# ╔═╡ ed5667e4-7691-451e-9c7b-4ddd367c4d48
vcov(m1; corr=true) # StatsBase.vcov: correlation matrix of fixed-effects coefficients

# ╔═╡ 973da4fc-3d95-469b-868b-33d933c43957
stderror(m1)          # StatsBase.stderror: SE for fixed-effects coefficients

# ╔═╡ da417df8-c674-4035-bb62-3c59ba18e349
propertynames(m1)  # names of available extractors

# ╔═╡ a55dc7ed-2e34-4fdf-a51b-b4c170538a89
md"""
## Parameter estimates of random-effect structure 

Aka: covariance parameter estimates
"""

# ╔═╡ e48923d7-be76-45bd-98de-cbecd82f4640
BlockDescription(m1) #  Description of blocks of A and L in a LinearMixedModel

# ╔═╡ 12ce94a6-4e04-40cd-bbad-2f5e8a766f77
VarCorr(m1) # MixedModels.VarCorr: estimates of random-effect structure (RES)

# ╔═╡ 55e8ac8e-2c95-4e96-921f-4a3a79fa17f9
#propertynames(m1)
m1.σ     # residual; m1.sigma, MixedModels.sdest(m1), sqrt(MixedModels.varest(m1))

# ╔═╡ 18daa693-e91f-4f72-a551-5c2e7d040de6
m1.σs    # VCs; m1.sigmas

# ╔═╡ 9ee525bb-a497-4796-94f2-418f81e8c53b
m1.θ     # Parameter vector for RES (w/o residual); m1.theta

# ╔═╡ 7f1928ee-91d8-41f9-8e1c-44c35fc33fd9
## check singularity
issingular(m1) # Test if model is singular for paramter vector m1.theta (default)

# ╔═╡ d1b850d4-9a24-4aae-abb6-69775d988044
md"## Effects PCA w/ CPs"

# ╔═╡ 8c5c4899-a95e-48a9-b5e3-8b7adf133bad
m1_pca=MixedModels.PCA(m1, corr=true)

# ╔═╡ b8173318-33b4-4e40-82ef-87ab43f92a03
md"""
# LMM with CPs for scores of Test (Table 3)

+ The scores LMM `m1` estimates CPs between scores of tests. 
+ This is a reparameterized version of the reference LMM `m1`.
"""

# ╔═╡ 4cbdd3b4-519d-437e-83ea-b949ebedaf8e
f1L = @formula zScore ~ 0 + Test * (age - 8.5) * Sex +
      (0+Test+(age-8.5)+Sex | School) +  (0+Test | Child) + (1 | Cohort);

# ╔═╡ 75337b24-3809-4130-bd21-7d36718c1ad1
m1L = fit(MixedModel, f1L, dat, contrasts=contr)

# ╔═╡ 81e6a46c-a252-4563-9b06-18295e060afa
m1L_pca=MixedModels.PCA(m1L, corr=true)

# ╔═╡ a4929cf8-0aa6-4c03-b95e-948c1fe72c20
md"""
- Seems to indicate that a1 term in School random effects can be removed
- Exposes a deficiency in the layout of the results in that `Test: Run` occurs in the random effects standard deviations but not in the fixed effects, so it is not in the table.
- Perhaps use `0 + ` in the fixed-effects too
"""

# ╔═╡ d5e71a82-1527-470b-99ea-5e7374331098
md"""
## Compare objective of m1 and m1L
 
m1L is a reparameterization of m1, so should be the same - in practice, should be close.
"""

# ╔═╡ 83f98594-6c29-4208-b956-7c98156ffb7e
(objective(m1), objective(m1L))

# ╔═╡ d0053d8a-6ef9-47c0-b5b5-18ba57503b5a
md"""
## Conditional modes of random effects

### Extraction
"""

# ╔═╡ 6a9657f3-0ab8-4622-8e82-ffaa2d98276a
begin
	cmL = raneftables(m1L);          # better: NamedTuple of columntables

	ChildL  = DataFrame(cmL.Child)
	Arrow.write("./data/fggk21_ChildL.arrow", ChildL, compress=:zstd)

	SchoolL = DataFrame(cmL.School)
	Arrow.write("./data/fggk21_SchoolL.arrow", SchoolL, compress=:zstd)
end

# ╔═╡ 2d315308-ade1-4bbd-8d2d-3946716b8bef
md"""
### Caterpillar plots


### Borrowing strengths (shrinkage)


# Various control LMMs a

## Age x Sex interactions nested in levels of Test

+ The nested LMM `m1` estimates the age x sex interaction for each (level of) test. 
+ This is a reparameterized version of the reference LMM `m1`.
"""

# ╔═╡ 1a2f9eb0-fa82-4e02-8d30-8dd420c5c161
f1_nested = @formula zScore ~ 0 + Test & ((age - 8.5)*Sex) +
            (0+Test+(age - 8.5)+Sex | School) +  (0+Test | Child) + (1 | Cohort);

# ╔═╡ 44b97bbe-0785-492b-908c-b0a687a8ff54
m1_nested = fit(MixedModel, f1_nested, dat, contrasts=contr)

# ╔═╡ 0edfc56d-939c-41f3-82b0-8e0af6e97bef
# compare objective - m1_nested is a reparameterization of m1
(m1.objective, m1_nested.objective)

# ╔═╡ 3fa58e82-d558-4974-b91f-26ba95c4c31d
md"""
None of the five interaction terms is significant.

## Quadratic trends of age

Check whether adding quadratic trends of age to the reference LMM increases the goodness of fit.
"""

# ╔═╡ 76af96e1-4157-40d3-a44e-193b74bf3f72
f1_agesq = @formula zScore ~ 1 + Test * ((age-8.5)+abs2(age-8.5)) * Sex  +
           (1+Test+(age-8.5)+Sex | School) +  (1+Test | Child) + (1 | Cohort);

# ╔═╡ ced3957b-ddcf-48cb-bfc3-c8c44749d966
m1_agesq = fit(MixedModel, f1_agesq, dat, contrasts=contr)

# ╔═╡ f10c2447-dfa5-4217-b2ff-7fdaac1bea92
MixedModels.likelihoodratiotest(m1, m1_agesq)

# ╔═╡ 15f75cc0-2135-45d4-b2e0-a834ac19710a
mods3 = [m1, m1_agesq];

# ╔═╡ 07755170-242d-47a6-a3b9-0efc3479f490
DataFrame(dof=dof.(mods3), deviance=deviance.(mods3), AIC=aic.(mods3),
	AICc=aicc.(mods3), BIC=bic.(mods3))

# ╔═╡ 90d22c96-178c-4754-8cb8-39bf1064bf3d
md"""
Adding quadratic trends of age to the fixed effects does not increase the goodness of fit.

# Observation level residuals for reference LMM m1

# Transfer (some) model objects to RCall
"""

# ╔═╡ c8a80ac2-af56-4503-b05d-d727d7bcac12


# ╔═╡ Cell order:
# ╠═d17d5cf8-988a-11eb-03dc-23f1449f5563
# ╟─6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
# ╠═4cd88e2b-c6f0-49cb-846a-eaae9f892e02
# ╠═d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
# ╠═8e122db8-416c-4ee7-b3c2-6e829659a606
# ╠═7dca92f9-dcc2-462c-b501-9ecabce74005
# ╟─d813057a-ddce-4bd5-bf40-f15b71c6eeed
# ╠═e41bb345-bd0a-457d-8d57-1e27dde43a63
# ╠═baf13d52-1569-4077-af4e-40aa53d38cf5
# ╟─153c3cb8-2488-4d70-9236-a5d4e47b6f1b
# ╠═9a3aa079-5bba-49ef-af84-6fd3e82a1ee7
# ╠═08763df8-aea2-4de7-bbb0-b4a1247c541a
# ╠═a230a7d8-d4a7-4d5a-8e02-69d14ee157fc
# ╠═dc38aff4-71a1-44e8-8154-a00efa9aa2cb
# ╟─60f522a1-27aa-48fb-af89-83cb2ce38cae
# ╠═0aac930e-4b2d-4b49-9590-1d21ca0b3217
# ╠═7cb282db-37dd-497f-9d73-edc563a105ef
# ╠═fb8f1bbf-7b7a-43b3-bedd-2977b533ffc4
# ╟─81decdea-8d98-453b-8d2b-a5ded6d815fd
# ╠═71c49ff4-e207-40f3-ba6c-ff5aa248557c
# ╠═534cc633-a02c-4ede-9d4b-d7ad46e38fd3
# ╠═804b8775-31ec-486f-88c6-0586d93c1666
# ╠═bc72326f-0ecf-44d1-ba4f-688ae7361662
# ╟─efaa575a-d03e-431b-a745-8553e095e111
# ╠═ee1cad9d-ebc8-4bb6-918a-9d38f0cd2232
# ╠═10c85aa3-9243-4c62-9041-90e78d27d3a9
# ╠═327f3fd1-0118-4952-8e0d-0779f69ad819
# ╟─2a5a7755-baf2-479e-9a57-4de137b5ecf3
# ╠═4c0a5cac-913d-490e-aeb9-505d34d89b5c
# ╠═8fbaf19e-869e-4cff-84a8-83023b572e87
# ╠═ce5d4ea2-e687-4bd0-8377-d3bf21f99721
# ╠═b955207b-d422-4579-b32c-e4273710d7ce
# ╠═98766877-fe20-4fac-a689-dd66d476cd40
# ╠═11291f17-1647-4b0f-8aad-26a24c75dfd1
# ╠═36c392c0-548a-4cc6-8786-34e626a71cc3
# ╠═7eb6ef49-ec67-4649-9d91-da4d90e82db7
# ╠═755ea5ac-bf09-4ead-9f00-0c1eb973554e
# ╠═bd69fae6-2d00-42e7-80f4-021c3638c043
# ╟─3838c638-9214-4ddd-b834-85dd1b2b1ce5
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
# ╟─d1b850d4-9a24-4aae-abb6-69775d988044
# ╠═8c5c4899-a95e-48a9-b5e3-8b7adf133bad
# ╟─b8173318-33b4-4e40-82ef-87ab43f92a03
# ╠═4cbdd3b4-519d-437e-83ea-b949ebedaf8e
# ╠═75337b24-3809-4130-bd21-7d36718c1ad1
# ╠═81e6a46c-a252-4563-9b06-18295e060afa
# ╟─a4929cf8-0aa6-4c03-b95e-948c1fe72c20
# ╟─d5e71a82-1527-470b-99ea-5e7374331098
# ╠═83f98594-6c29-4208-b956-7c98156ffb7e
# ╟─d0053d8a-6ef9-47c0-b5b5-18ba57503b5a
# ╠═6a9657f3-0ab8-4622-8e82-ffaa2d98276a
# ╟─2d315308-ade1-4bbd-8d2d-3946716b8bef
# ╠═1a2f9eb0-fa82-4e02-8d30-8dd420c5c161
# ╠═44b97bbe-0785-492b-908c-b0a687a8ff54
# ╠═0edfc56d-939c-41f3-82b0-8e0af6e97bef
# ╟─3fa58e82-d558-4974-b91f-26ba95c4c31d
# ╠═76af96e1-4157-40d3-a44e-193b74bf3f72
# ╠═ced3957b-ddcf-48cb-bfc3-c8c44749d966
# ╠═f10c2447-dfa5-4217-b2ff-7fdaac1bea92
# ╠═15f75cc0-2135-45d4-b2e0-a834ac19710a
# ╠═07755170-242d-47a6-a3b9-0efc3479f490
# ╟─90d22c96-178c-4754-8cb8-39bf1064bf3d
# ╠═c8a80ac2-af56-4503-b05d-d727d7bcac12
