### A Pluto.jl notebook ###
# v0.15.1

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

# ╔═╡ 3c7a1a36-fa48-11eb-00d9-9f2255af9992
begin
	using CairoMakie
	using DataFrames
	using MixedModels
	using NLopt
	using ProgressMeter
	using PlutoUI
	using StatsBase
	
	using MixedModels: dataset, _check_nlopt_return, updateL!, setθ!, nθ
end

# ╔═╡ f9e37852-7116-4af6-a61d-2ea03b1ba4ac
StatsBase.coefnames(re::MixedModels.AbstractReMat) = re.cnames

# ╔═╡ 1fa2d657-5506-436a-a25a-5203c20e510a
begin
	mrk17 = MixedModels.dataset(:mrk17_exp1)
	contr = Dict(:subj => Grouping(),
				 :item => Grouping(),
				 :F => HelmertCoding(),
			     :P => HelmertCoding(),
			 	 :Q => HelmertCoding(),
				 :lQ => HelmertCoding(),
				 :lT => HelmertCoding())
	m6frm = @formula(1000/rt ~ 1 + F * P * Q * lQ * lT + 
		                      (1 +     P + Q + lQ + lT| item) + 
		                      (1 + F + P + Q + lQ +lT | subj))
	nothing
end

# ╔═╡ 9f8ca9f2-ebf4-4b40-a331-9cc4e4ba0bc6
function logged_fit!(m::LinearMixedModel{T}, log::Vector{<:Tuple}; progress::Bool=true, REML::Bool=false) where {T}
    optsum = m.optsum
    MixedModels.unfit!(m)
    opt = Opt(optsum)
    optsum.REML = REML
    prog = ProgressUnknown("Minimizing"; showspeed=true)
	empty!(log)
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = objective(updateL!(setθ!(m, x)))
		push!(log, (copy(x), val))
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val)])
        return val
    end
    NLopt.min_objective!(opt, obj)
    optsum.finitial = obj(optsum.initial, T[])
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    ProgressMeter.finish!(prog)
    ## check if small non-negative parameter values can be set to zero
    xmin_ = copy(xmin)
    lb = optsum.lowerbd
    for i in eachindex(xmin_)
        if iszero(lb[i]) && zero(T) < xmin_[i] < T(0.001)
            xmin_[i] = zero(T)
        end
    end
    if xmin_ ≠ xmin
        if (zeroobj = obj(xmin_, T[])) ≤ (fmin + 1.e-5)
            fmin = zeroobj
            copyto!(xmin, xmin_)
        end
    end
    ## ensure that the parameter values saved in m are xmin
    updateL!(setθ!(m, xmin))

    optsum.feval = opt.numevals
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    _check_nlopt_return(ret)
    return m
end

# ╔═╡ 2c3990e0-0253-4c4d-99ed-69636e7cf4e7
@bind frate PlutoUI.Slider(1:100; default=10, show_value=true)

# ╔═╡ b80d463a-cb34-49f9-ab69-2abeb9b629e9
models = Dict("sleepstudy" => LinearMixedModel(@formula(reaction ~ 1 + days + (1 + days|subj)), dataset(:sleepstudy)),
		     "mrk17" => LinearMixedModel(m6frm, mrk17; contrasts=contr),
			 "kb07" => LinearMixedModel(@formula(rt_trunc ~ 1+spkr*prec*load+(1+spkr+prec+load|subj)+(1+spkr+prec+load|item)), dataset(:kb07)),
			"kb07_int" => LinearMixedModel(@formula(rt_trunc ~ 1+spkr*prec*load+(1|subj)+(1|item)), dataset(:kb07)));

# ╔═╡ c259a93c-4c42-4708-8a57-6ea312aedc04
@bind mname PlutoUI.Select(collect(keys(models)); default="sleepstudy")

# ╔═╡ f0e95022-ee10-470b-9753-53d2510309ae
begin
	m = models[mname]
	l = [(copy(m.optsum.initial), m.optsum.finitial)]
	logged_fit!(m, l)
	m
end

# ╔═╡ 08efd2c8-faaf-4be2-99aa-a3c3d41166de
m.optsum

# ╔═╡ 844ec630-e50e-4ccd-bb0a-265777be21bb
@bind grp PlutoUI.Select(collect(string.(fnames(m))))

# ╔═╡ c42c9c10-798c-46c9-9013-0cd9a092cd10
path = mktempdir()

# ╔═╡ 42726a44-4723-4d9f-954e-79635ea0d365
llvid = let path = path, fig = Figure(), frate = frate, l = l

	pp = Node(Point2f0[])
    ax = Axis(fig[1, 1])
	
	lines!(ax, pp)
	ax.xlabel = "Iteration"
	ax.ylabel = "-2 Log likelihood"
	
	
	record(fig, joinpath(path, "ll_animation.mp4"), enumerate(l), framerate = frate) do (idx, (θ, ll))
		push!(pp[], Point2f0(idx, ll))
		autolimits!(ax)
	end
end

# ╔═╡ 7a69147b-0b13-4339-9d53-9748ed506c03
LocalResource(llvid)

# ╔═╡ 96affd87-52ee-4eff-8f8a-7099fe7666e1
shrinkagevid = let path = path, fig = Figure(; resolution=(700, 700)), frate = frate, m = m, gf = grp, θref = 10000 * m.optsum.initial
	
	gf = Symbol(gf)
	#gf = :item
	reind = findfirst(==(gf), fnames(m))
	
	if isnothing(reind)
        throw(ArgumentError("gf=$gf is not one of the grouping factor names, $(fnames(m))"))
    end
	
	reest = ranef(updateL!(setθ!(m, m.optsum.final)))[reind] 
	reref = ranef(updateL!(setθ!(m, θref)))[reind]    
    cnms = m.reterms[reind].cnames
    k = size(reest, 1)
	cols = Dict()
	uvidx = 0
	uv = Node[]
	
    for i in 2:k                          # strict lower triangle of panels
		uvidx += 1
		row = Axis[]
        for j in 1:(i - 1)
            ax = Axis(fig[i - 1, j]; aspect=AxisAspect(1))
			push!(row, ax)
			col = get!(cols, j, Axis[])
			push!(col, ax)
            xy = Node(Point2f0.(view(reref, j, :), view(reref, i, :)))
            # reference points
			scatter!(ax, xy; color=(:red, 0.25))
            uvpp = Node(Point.(view(reref, j, :), view(reref, i, :)))
			push!(uv, uvpp)
			# first so arrow heads don't obscure pts
			movement = lift(uvpp) do vals
			 	Point.(first.(vals) .- first.(xy[]), last.(vals) .-  last.(xy[]))
			end
			arrows!(ax, xy, movement)        
			# conditional means at estimates
			scatter!(ax, uvpp; color=(:blue, 0.25))  #
            if i == k              # add x labels on bottom row
                ax.xlabel = string(cnms[j])
            else
                hidexdecorations!(ax; grid=false)
            end
            if isone(j)            # add y labels on left column
                ax.ylabel = string(cnms[i])
            else
                hideydecorations!(ax; grid=false)
            end
        end
		linkyaxes!(row...)
    end
	
	foreach(values(cols)) do col
		linkxaxes!(col...)
	end	
	record(fig, joinpath(path, "shrinkage_animation.mp4"), l, framerate = frate) do (θ, ll)
		reest = ranef(updateL!(setθ!(m, θ)))[reind]
		uvidx = 0
		for i in 2:k, j in 1:(i - 1)
			uvidx += 1
			uv[uvidx][] = Point.(view(reest, j, :), view(reest, i, :))
		end
	end
end

# ╔═╡ fb767203-4f5f-48fc-acce-e235f7e79868
LocalResource(shrinkagevid)

# ╔═╡ 2fbf6c68-7fe7-4162-8087-4c6184644274
VarCorr(m)

# ╔═╡ d5df3d20-dbf5-4d12-a87e-9a2d7e3f78d0
bouncyθvid = let path = path, fig = Figure(), frate = 5, l = l, m = m
	
	supertitle = Node("-2 log likelihood: ")
	l1 = Label(fig[1, 1], supertitle, textsize = 15)
	l1.tellwidth = false
	itertitle = Node("iter:")
	l2 = Label(fig[1, 2], itertitle, textsize = 15)
	l2.tellwidth = false
	titlelayout = GridLayout()
	fig[1, 1:2] = titlelayout
	
	axpar = Axis(fig[2, :])
	axpar.tellwidth = true
	θs = Node(copy(m.optsum.initial))
	xs = 1:length(θs[])
	iter = Node(1.0)
	maxiter = length(l)
	idx = 1
	color = lift(iter) do iter
		(:black, 0.75 ^ (iter - idx))
	end
	scatter!(axpar, xs, θs; color)
	lines!(axpar, xs, θs; color)
	autolimits!(axpar)
	
	θnames = sizehint!(String[], sum(nθ, m.reterms))
	for (g, gname) in enumerate(fnames(m))
		cnms = coefnames(m.reterms[g])
		for i in 1:length(cnms), j in 1:i
			if i == j
				push!(θnames, string("σ_", gname, "_", cnms[i]))
			else
				push!(θnames, string("ρ_", gname, "_", cnms[j], "_", cnms[i]))
			end		
		end
	end
 	axpar.xticks[] = (xs, θnames)
	axpar.xticklabelrotation[] = pi/4
	axpar.xlabel = "θ components"
	axpar.ylabel = "θ values"
	
	vid = record(fig, joinpath(path, "bouncy_theta_animation.mp4"), enumerate(l), framerate = frate) do (idx, (θ, ll))
		iter[] = idx
		color = lift(iter) do iter
			(:black, 0.75 ^ (iter - idx))
		end
		scatter!(axpar, xs, θ; color)
		lines!(axpar, xs, θ; color)
		supertitle[] = "-2 log likelihood: $(round(Int,ll))"
		itertitle[] = "iter: $(idx)"
		autolimits!(axpar)
	end
	vid
end

# ╔═╡ b265d9ec-ad73-4a8d-aeba-b765b2ea9bcd
LocalResource(bouncyθvid)

# ╔═╡ fc29e293-8eab-4f3b-99c3-4ae11389f881
θvid = let path = path, fig = Figure(), frate = frate, l = l
	
	ppll = Node(Point2f0[])
    axll = Axis(fig[1, 1])
	lines!(axll, ppll)
	axll.ylabel = "-2 log likelihood"
	hidexdecorations!(axll)
	
	#grplens 
	
	# coridx = findall(==(-Inf), m.lowerbd)
	# vardidx = findall(==(0), m.lowerbd)
	
	ppθ = Node(zeros(Float32, length(m.θ), 0))
	isvar = Int.(m.θ .!= 0)
	grps = mapreduce(vcat, enumerate(m.reterms)) do (idx, re)
		idx * ones(Int, MixedModels.nθ(re))
	end
	cols = cgrad(:Dark2_8; categorical=unique(grps), alpha=0.6)[grps]
	labs = string.(MixedModels.fname.(m.reterms[grps]))
	
	axθ = Axis(fig[2, 1])
	s = series!(axθ, ppθ; solid_color=cols, labels=labs)
	axislegend(axθ; unique=true)
	axθ.xlabel = "Iteration"
	axθ.ylabel = "θ"
	linkxaxes!(axll, axθ)
	
	vid = record(fig, joinpath(path, "fit_animation.mp4"), enumerate(l), framerate = frate) do (idx, (θ, ll))
		push!(ppll[], Point2f0(idx, ll))
		ppθ[] = hcat(ppθ[], reshape(θ, :, 1))
		autolimits!(axθ)
		autolimits!(axll)
	end
	vid
end

# ╔═╡ d706802d-7df1-41cf-81f0-8856718ea10a
LocalResource(θvid)

# ╔═╡ 1f537c5f-1da2-4f21-961c-6a27c607404a
llshrinkagevid = let path = path, fig = Figure(; resolution=(1200, 1000)), frate = frate, m = m, gf = grp, θref = 10000 * m.optsum.initial
	
	gf = Symbol(gf)
	# gf = :item
	reind = findfirst(==(gf), fnames(m))
	
	if isnothing(reind)
        throw(ArgumentError("gf=$gf is not one of the grouping factor names, $(fnames(m))"))
    end
	
	reest = ranef(updateL!(setθ!(m, m.optsum.final)))[reind] 
	reref = ranef(updateL!(setθ!(m, θref)))[reind]    
    cnms = coefnames(m.reterms[reind])
    k = size(reest, 1)
	cols = Dict()
	uvidx = 0
	uv = Node[]
	θnames = String[]
	
    for i in 2:k                          # strict lower triangle of panels
		uvidx += 1
		row = Axis[]
        for j in 1:(i - 1)
            ax = Axis(fig[i - 1, j])
			push!(row, ax)
			col = get!(cols, j, Axis[])
			push!(col, ax)
            xy = Node(Point2f0.(view(reref, j, :), view(reref, i, :)))
            # reference points
			scatter!(ax, xy; color=(:red, 0.25))
            uvpp = Node(Point.(view(reref, j, :), view(reref, i, :)))
			push!(uv, uvpp)
			# first so arrow heads don't obscure pts
			movement = lift(uvpp) do vals
			 	Point.(first.(vals) .- first.(xy[]), last.(vals) .-  last.(xy[]))
			end
			arrows!(ax, xy, movement)        
			# conditional means at estimates
			scatter!(ax, uvpp; color=(:blue, 0.25))  #
            if i == k              # add x labels on bottom row
                ax.xlabel = string(cnms[j])
            else
                hidexdecorations!(ax; grid=false)
            end
            if isone(j)            # add y labels on left column
                ax.ylabel = string(cnms[i])
            else
                hideydecorations!(ax; grid=false)
            end
			
        end
		linkyaxes!(row...)
    end
	
	foreach(values(cols)) do col
		linkxaxes!(col...)
	end
	
	ppll = Node(Point2f0[])
    axll = Axis(fig[k, 1:k])
	lines!(axll, ppll)
	axll.ylabel = "-2 log likelihood"
	hidexdecorations!(axll)
	
	
	ppθ = Node(zeros(Float32, nθ(m.reterms[reind]), 0))
	θstart = reind == 1 ? 1 : sum(nθ, m.reterms[1:(reind-1)]) + 1
	θind = θstart:(θstart + nθ(m.reterms[reind]) - 1)
	
	θnames = sizehint!(String[], nθ(m.reterms[reind]))
	cnms = coefnames(m.reterms[reind])
	for i in 1:length(cnms), j in 1:i
		if i == j
			push!(θnames, string("σ_", cnms[i]))
		else
			push!(θnames, string("ρ_", cnms[j], "_", cnms[i]))
		end		
	end
	
	axθ = Axis(fig[k+1, 1:k])
	kwargs = if length(θnames) < 10
		(; color=:Paired_10, labels=θnames)
	else
		(; solid_color=(:black, 0.6))
	end
	s = series!(axθ, ppθ; kwargs...)
	# axislegend(axθ; unique=true)
	axθ.xlabel = "Iteration"
	axθ.ylabel = "θ"
	linkxaxes!(axll, axθ)
		
	record(fig, joinpath(path, "llshrinkage_animation.mp4"), enumerate(l), framerate = frate) do (idx, (θ, ll))
		reest = ranef(updateL!(setθ!(m, θ)))[reind]
		uvidx = 0
		for i in 2:k, j in 1:(i - 1)
			uvidx += 1
			uv[uvidx][] = Point.(view(reest, j, :), view(reest, i, :))
		end
		push!(ppll[], Point2f0(idx, ll))
		ppθ[] = hcat(ppθ[], reshape(view(θ, θind), :, 1))
		autolimits!(axll)
		autolimits!(axθ)
	end
end

# ╔═╡ 66fce748-8bf5-4ff7-8644-0d6293e08dda
LocalResource(llshrinkagevid)

# ╔═╡ 2ae64f75-2fdf-4efd-9228-8e4efd49bfc8
bouncyshrinkagevid = let path = path, fig = Figure(; resolution=(1200, 1000)), frate = frate, m = m, gf = grp, θref = 10000 * m.optsum.initial
	
	gf = Symbol(gf)
	# gf = :item
	reind = findfirst(==(gf), fnames(m))
	
	if isnothing(reind)
        throw(ArgumentError("gf=$gf is not one of the grouping factor names, $(fnames(m))"))
    end
	
	supertitle = Node("-2 log likelihood: ")
	# Label(fig[0, 1], supertitle, textsize = 15)
	
	reest = ranef(updateL!(setθ!(m, m.optsum.final)))[reind] 
	reref = ranef(updateL!(setθ!(m, θref)))[reind]    
    cnms = coefnames(m.reterms[reind])
    k = size(reest, 1)
	cols = Dict()
	uvidx = 0
	uv = Node[]
	θnames = String[]
	
    for i in 2:k                          # strict lower triangle of panels
		uvidx += 1
		row = Axis[]
        for j in 1:(i - 1)
            ax = Axis(fig[i - 1, j])
			push!(row, ax)
			col = get!(cols, j, Axis[])
			push!(col, ax)
            xy = Node(Point2f0.(view(reref, j, :), view(reref, i, :)))
            # reference points
			scatter!(ax, xy; color=(:red, 0.25))
            uvpp = Node(Point.(view(reref, j, :), view(reref, i, :)))
			push!(uv, uvpp)
			# first so arrow heads don't obscure pts
			movement = lift(uvpp) do vals
			 	Point.(first.(vals) .- first.(xy[]), last.(vals) .-  last.(xy[]))
			end
			arrows!(ax, xy, movement)        
			# conditional means at estimates
			scatter!(ax, uvpp; color=(:blue, 0.25))  #
            if i == k              # add x labels on bottom row
                ax.xlabel = string(cnms[j])
            else
                hidexdecorations!(ax; grid=false)
            end
            if isone(j)            # add y labels on left column
                ax.ylabel = string(cnms[i])
            else
                hideydecorations!(ax; grid=false)
            end
			
        end
		linkyaxes!(row...)
    end
	
	foreach(values(cols)) do col
		linkxaxes!(col...)
	end
	
	
	ppll = Node(Point2f0[])
    axll = Axis(fig[k, 1:k])
	lines!(axll, ppll)
	axll.ylabel = "-2 log likelihood"
	axll.xlabel = "iteration"
	
	axpar = Axis(fig[(k+1):(k+1), 1:k])
	θs = Node(copy(m.optsum.initial))
	xs = 1:length(θs[])
	iter = Node(1.0)
	maxiter = length(l)
	idx = 1
	color = lift(iter) do iter
		(:black, 0.75 ^ (iter - idx))
	end
	scatter!(axpar, xs, θs; color)
	lines!(axpar, xs, θs; color)
	autolimits!(axpar)
	
	θnames = sizehint!(String[], sum(nθ, m.reterms))
	for (g, gname) in enumerate(fnames(m))
		cnms = coefnames(m.reterms[g])
		for i in 1:length(cnms), j in 1:i
			if i == j
				push!(θnames, string("σ_", gname, "_", cnms[i]))
			else
				push!(θnames, string("ρ_", gname, "_", cnms[j], "_", cnms[i]))
			end		
		end
	end
 	axpar.xticks[] = (xs, θnames)
	axpar.xticklabelrotation[] = pi/4
	axpar.xlabel = "θ components"
	axpar.ylabel = "θ values"
	
	vid = record(fig, joinpath(path, "bouncy_shrinkage_animation.mp4"), enumerate(l), framerate = frate) do (idx, (θ, ll))
		
		reest = ranef(updateL!(setθ!(m, θ)))[reind]
		uvidx = 0
		for i in 2:k, j in 1:(i - 1)
			uvidx += 1
			uv[uvidx][] = Point.(view(reest, j, :), view(reest, i, :))
		end
		
		push!(ppll[], Point2f0(idx, ll))
		autolimits!(axll)
		
		iter[] = idx
		color = lift(iter) do iter
			(:black, 0.75 ^ (iter - idx))
		end
		scatter!(axpar, xs, θ; color)
		lines!(axpar, xs, θ; color)
		supertitle[] = "-2 log likelihood: $(round(ll))"
		autolimits!(axpar)

	end
	vid
end

# ╔═╡ 1540378e-664c-4f51-a641-35bc98eefc8e
LocalResource(bouncyshrinkagevid)

# ╔═╡ 23ef35f6-cfb5-422d-a7af-e9af7a5520b5
fitvid = let path = path, fig = Figure(; resolution=(700, 700)), frate = frate, l = l, m = m
	supertitle = Node("-2 log likelihood: ")
	limits = [0.9, 1.1] .* extrema(response(m))
	observed = Node(response(m))
	predicted = Node(ones(size(observed[])) * mean(observed[]))
    ax = Axis(fig[1, 1:2]; aspect=AxisAspect(1))
	Label(fig[0, 1], supertitle, textsize = 15)
	
	scatter!(ax, observed, predicted; color=(:steelblue, 0.3))
	ax.xlabel = "Observed"
	ax.ylabel = "Fitted"
	abline!(ax, 0, 1; linestyle=:dash, color=(:black, 1.0))
	xlims!(limits...)
	ylims!(limits...)
	
	record(fig, joinpath(path, "fit_pred.mp4"), enumerate(l), framerate = frate) do (idx, (θ, ll))
		predicted[] = fitted(updateL!(setθ!(m, θ)))
		supertitle[] = "-2 log likelihood: $(round(Int,ll))"
	end
end

# ╔═╡ 080d9245-3f7b-4154-8d50-c1ca338d3ffb
LocalResource(fitvid)

# ╔═╡ c23c46cf-2963-4009-9b02-fc11772fcc88
resvid = let path = path, fig = Figure(), frate = frate, l = l, m = m
	supertitle = Node("-2 log likelihood: ")
	Label(fig[1, 1], supertitle, textsize = 15)
	
	limits = [0.9, 1.1] .* extrema(response(m))
	res = Node(zeros(Float32, nobs(m)))
	predicted = Node(ones(size(res[])) * mean(res[]))
    ax = Axis(fig[2, 1:2])
	
	scatter!(ax, predicted, res; color=(:steelblue, 0.3))
	ax.xlabel = "Fitted"
	ax.ylabel = "Residual"
	hlines!(ax, [0]; linestyle=:dash, color=(:black, 1.0))
	xlims!(limits...)
	
	record(fig, joinpath(path, "res_pred.mp4"), enumerate(l), framerate = frate) do (idx, (θ, ll))
		predicted[] = fitted(updateL!(setθ!(m, θ)))
		res[] = residuals(m)
		supertitle[] = "-2 log likelihood: $(round(Int,ll))"
		autolimits!(ax) # can't find a way to do this just for y
		xlims!(limits...)
	end
end

# ╔═╡ c9c4b58f-01dc-46d0-8a0b-6191c4f17ecf
LocalResource(resvid)

# ╔═╡ 7dddde1b-0e2a-4972-81c9-85f6790e27f1
qqvid = let path = path, fig = Figure(; resolution=(700, 700)), frate = frate, l = l, m = m
	supertitle = Node("-2 log likelihood: ")
	Label(fig[1, 1], supertitle, textsize = 15)
	
	res = Node(zeros(Float32, nobs(m)))
    ax = Axis(fig[2, 1:2]; aspect=AxisAspect(1))
	
	qqnorm!(ax, res; alpha=(:steelblue, 0.3))
	ax.xlabel = "Standard Normal Quantiles"
	ax.ylabel = "Standardized Observed Quantiles"

	
	record(fig, joinpath(path, "res_pred.mp4"), enumerate(l), framerate = frate) do (idx, (θ, ll))
		res[] = residuals(updateL!(setθ!(m, θ)))
		supertitle[] = "-2 log likelihood: $(round(ll))"
		autolimits!(ax)
	end
end

# ╔═╡ 07d07f17-55ec-4092-8183-48d9dca4b21e
LocalResource(qqvid)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MixedModels = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
NLopt = "76087f3c-5699-56af-9a33-bf431cd00edd"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CairoMakie = "~0.6.2"
DataFrames = "~1.2.2"
MixedModels = "~4.0.0"
NLopt = "~0.6.0"
PlutoUI = "~0.7.4"
ProgressMeter = "~1.7.1"
StatsBase = "~0.33.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "2e004e61f76874d153979effc832ae53b56c20ee"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.22"

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

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "f0853a9d2cf2cf4f9fbd98c6e19ff1627e65a9a7"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.2.4"

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
git-tree-sha1 = "68628add03c2c7c2235834902aad96876f07756a"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.6.2"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "59fe0cb37784288d6b9f1baebddbf75457395d40"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.0"

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
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random", "StaticArrays"]
git-tree-sha1 = "ed268efe58512df8c7e224d2e170afd76dd6a417"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.13.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[ColorVectorSpace]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "StatsBase"]
git-tree-sha1 = "4d17724e99f357bfd32afa0a9e2dda2af31a9aea"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.8.7"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

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
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

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
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3889f646423ce91dd1055a76317e9a1d3a23fff1"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.11"

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
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

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
git-tree-sha1 = "f985af3b9f4e278b1d24434cbb546d6092fca661"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.3"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3676abafff7e4ff07bbd2c42b3d8201f31653dcc"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.9+8"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "256d8e6188f3f1ebfa1a5d17e072a0efafa8c5bf"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.10.1"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8c8eac2af06ce35973c3eadb4ab3243076a408e7"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.1"

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
git-tree-sha1 = "d51e69f0a2f8a3842bca4183b700cf3d9acce626"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.1"

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

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "4136b8a5668341e58398bb472754bff4ba0456ff"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.3.12"

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
git-tree-sha1 = "d44945bdc7a462fa68bb847759294669352bd0a4"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.5.7"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "PNGFiles"]
git-tree-sha1 = "0d6d09c28d67611c68e25af0c2df7269c82b73c7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.4.1"

[[IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "1e0e51692a3a77f1eeb51bf741bdd0439ed210e7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.2"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

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
deps = ["Dates", "Mmap", "Unicode"]
git-tree-sha1 = "565947e5338efe62a7db0aa8e5de782c623b04cd"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.20.1"

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
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "c253236b0ed414624b083e6b72bfe891fbd2c7af"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+1"

[[Makie]]
deps = ["Animations", "Artifacts", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LinearAlgebra", "MakieCore", "Markdown", "Match", "Observables", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "82f9d9de6892e5470372720d23913d14a08991fd"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.14.2"

[[MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "7bcc8323fb37523a6a51ade2234eee27a11114c8"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.1.3"

[[MappedArrays]]
git-tree-sha1 = "18d3584eebc861e311a552cbb67723af8edff5de"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Match]]
git-tree-sha1 = "5cf525d97caf86d29307150fcba763a64eaa9cbe"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.1.0"

[[MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[MixedModels]]
deps = ["Arrow", "DataAPI", "Distributions", "GLM", "JSON3", "LazyArtifacts", "LinearAlgebra", "Markdown", "NLopt", "PooledArrays", "ProgressMeter", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StatsModels", "StructTypes", "Tables"]
git-tree-sha1 = "c17a2cdb285cf936920855e48e198e4f25fcef5c"
uuid = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
version = "4.0.0"

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

[[NLopt]]
deps = ["MathProgBase", "NLopt_jll"]
git-tree-sha1 = "48d523294d66f34d012e224ec3082d35c395ebd2"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.0"

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
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "5cc97a6f806ba1b36bac7078b866d4297ae8c463"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.4"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

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
git-tree-sha1 = "520e28d4026d16dcf7b8c8140a3041f0e20a9ca8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.7"

[[Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "f4049d379326c2c7aa875c702ad19346ecb2b004"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.1"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fa5e78929aebc3f6b56e1a88cf505bb00a354c4"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.8"

[[Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9bc1871464b12ed19297fbc56c4fb4ba84988b0d"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.47.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "477bf42b4d1496b454c10cce46645bb5b8a0cf2c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "8b7989f1918a5ea044e05d8e2012822c9e63df4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.4"

[[PolygonOps]]
git-tree-sha1 = "c031d2332c9a8e1c90eca239385815dc271abb22"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.1"

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

[[Ratios]]
git-tree-sha1 = "37d210f612d70f3f7d57d488cb3b6eff56ad4e41"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.0"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

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

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "a3a337914a035b2d59c9cbe7f1a38aaba1265b02"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.6"

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
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "508822dca004bf62e210609148511ad03ce8f1d8"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.0"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "62701892d172a2fa41a1f829f66d2b0db94a9a63"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3fedeffc02e47d6e3eb479150c8e5cd8f15a77a0"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.10"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "a209a68f72601f8aa0d3a7c4e50ba3f67e32e6f8"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.24"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "Tables"]
git-tree-sha1 = "44b3afd37b17422a62aea25f04c1f7e09ce6b07f"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.5.1"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "e36adc471280e8b346ea24c5c87ba0571204be7a"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.2"

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
git-tree-sha1 = "81753f400872e5074768c9a77d4c44e70d409ef0"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.6"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "7c53c35547de1c5b9d46a4797cf6d8253807108c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.5"

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
# ╠═3c7a1a36-fa48-11eb-00d9-9f2255af9992
# ╠═f9e37852-7116-4af6-a61d-2ea03b1ba4ac
# ╠═1fa2d657-5506-436a-a25a-5203c20e510a
# ╠═9f8ca9f2-ebf4-4b40-a331-9cc4e4ba0bc6
# ╠═2c3990e0-0253-4c4d-99ed-69636e7cf4e7
# ╟─b80d463a-cb34-49f9-ab69-2abeb9b629e9
# ╟─c259a93c-4c42-4708-8a57-6ea312aedc04
# ╠═f0e95022-ee10-470b-9753-53d2510309ae
# ╠═08efd2c8-faaf-4be2-99aa-a3c3d41166de
# ╟─844ec630-e50e-4ccd-bb0a-265777be21bb
# ╠═c42c9c10-798c-46c9-9013-0cd9a092cd10
# ╠═42726a44-4723-4d9f-954e-79635ea0d365
# ╠═7a69147b-0b13-4339-9d53-9748ed506c03
# ╠═96affd87-52ee-4eff-8f8a-7099fe7666e1
# ╠═fb767203-4f5f-48fc-acce-e235f7e79868
# ╠═2fbf6c68-7fe7-4162-8087-4c6184644274
# ╠═d5df3d20-dbf5-4d12-a87e-9a2d7e3f78d0
# ╠═b265d9ec-ad73-4a8d-aeba-b765b2ea9bcd
# ╠═fc29e293-8eab-4f3b-99c3-4ae11389f881
# ╠═d706802d-7df1-41cf-81f0-8856718ea10a
# ╠═1f537c5f-1da2-4f21-961c-6a27c607404a
# ╠═66fce748-8bf5-4ff7-8644-0d6293e08dda
# ╠═2ae64f75-2fdf-4efd-9228-8e4efd49bfc8
# ╠═1540378e-664c-4f51-a641-35bc98eefc8e
# ╠═23ef35f6-cfb5-422d-a7af-e9af7a5520b5
# ╠═080d9245-3f7b-4154-8d50-c1ca338d3ffb
# ╠═c23c46cf-2963-4009-9b02-fc11772fcc88
# ╠═c9c4b58f-01dc-46d0-8a0b-6191c4f17ecf
# ╠═7dddde1b-0e2a-4972-81c9-85f6790e27f1
# ╠═07d07f17-55ec-4092-8183-48d9dca4b21e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
