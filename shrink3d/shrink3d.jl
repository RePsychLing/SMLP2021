using GLMakie
using MixedModels
using MixedModels: dataset
using MixedModelsMakie

# we scale weirdly here so that we have nice scaling in the 3d plot
m = fit(MixedModel,
        @formula(rt_trunc / 100 ~ 1 + spkr*prec*load + (1|subj) + (1+spkr+prec|item)),
        dataset(:kb07);
        contrasts=Dict(:load => HelmertCoding(),
                        :spkr => HelmertCoding(),
                        :prec => HelmertCoding()))

shrinkageplot(m, :item)

let gf = :item, θref = 10000*m.optsum.initial
    reind = findfirst(==(gf), fnames(m))  # convert the symbol gf to an index
    if isnothing(reind)
        throw(ArgumentError("gf=$gf is not one of the grouping factor names, $(fnames(m))"))
    end
    reest = ranef(m)[reind]               # random effects conditional means at estimated θ
    reref = ranef(updateL!(setθ!(m, θref)))[reind]  # same at θref
    updateL!(setθ!(m, m.optsum.final))    # restore parameter estimates and update m

    tails = [Point3f(reref[:, j]) for j in 1:size(reref,2)]
    lengths = [Vec3f(reest[:, j] .- reref[:, j]) for j in 1:size(reref,2)]
    f = Figure()
    ax3 = Axis3(f[1,1])
    arrows!(ax3, tails, lengths)
    ax3.xlabel = "Intercept"
    ax3.ylabel = "spkr"
    ax3.zlabel = "prec"
    f
end
