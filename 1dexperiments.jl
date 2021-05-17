using LinearAlgebra
using HCubature
using SpecialFunctions
using PyPlot
using SparseArrays
using ProgressBars
using StatsBase
using AlphaStableDistributions
using AdvancedHMC
using Dates
using MAT
using Optim
using MCMCDiagnostics
using Dates


function difference1(X)
    N = length(X);
    M = zeros(N,N);

    for i  = 2:N
       M[i,i] = -1;
       M[i,i-1] = 1;
    end
    M[1,1] = 1;
    return M
end

function  difference2(X)
    N = length(X);
    M = zeros(N,N);

    for i  = 3:N
       M[i,i] = -1;
       M[i,i-1] = 2;
       M[i,i-2] = -1;
    end
    M[1,1] = 1; M[2,1] = 1; M[2,2] = -1;
    return M

end

function smatrix(X0,a,b)
    N = length(X0);
    dx = X0[2]-X0[1];

    M = zeros(N,N);
    for i=2:N-1
        M[i,i-1] = -a/dx^2;
        M[i,i+1] = -a/dx^2;
        M[i,i] = 2*a/dx^2 +b;

    end
    #M[1,:] = [b+2*a/dx^2 -a/dx^2 zeros(1,N-3) -a/dx^2];
    #M[N,:] = [-a/dx^2 zeros(1,N-3) -a/dx^2 b+2*a/dx^2];

    M[1,1] = 1;
    #M(N,1) = 1/dx;
    #M(N,2) = -1/dx;

    M[N,N] = 1
    return M

end

function measurementmatrix(X,MX,kernel)
    s = X[2]-X[1];
    F = zeros(size(MX)[1],size(X)[1])
    for i = 1:size(MX)[1]
        F[i,:] = s.*kernel.(X,MX[i]);
    end
    return F
end

function divide(N,NT)
    even = mod(N,NT) == 0
    k = Int64(floor(N/NT))
    t = 0
    if(even)
        q = Vector{Vector{Int64}}(undef,k)
        t = k

    else
        q = Vector{Vector{Int64}}(undef,k+1)
        t = k +1
    end

    for i = 1:t
        q[i] = Vector(NT*(i-1)+1:min(NT*i,N))
    end

    return q


end


function partialreplace(orig,replace,ix)
    a = copy(orig)
    a[ix] = replace
    return a
end

function  logpcauchy(W,arguments)
    Mat = arguments.Mat; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; #F = arguments.F
    G = 1;

    r = Mat*(W);
    #r = F*U;

    logp = -0.5/noisesigma^2*sum((r-y).^2);
    logp = logp + sum(log.(scale./(pi*(scale^2.0.+W.^2))));

    return logp;

end

function logpcauchygradi(W,arguments;both=false)
    logp = 0.0
    Mat = arguments.Mat; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; #F = arguments.F

    r = Mat*(W);
    #r = F*U;

    if(both)
        logp = -0.5/noisesigma^2*sum((r-y).^2);
        logp = logp + sum(log.(scale./(pi*(scale^2.0.+W.^2))));
    end

    Gr = Mat

    nt = length(y); N =size(W)[1];
    G = zeros(N,);
    G =  -1/noisesigma^2*Mat'*(r-y)
    # for i = 1:nt
    #     q =(-((r[i]-y[i])./noisesigma.^2))*Gr[i,:]';
    #     G = G + dropdims(q, dims = tuple(findall(size(q) .== 1)...))
    # end

   G = G  -2*W./(scale^2 .+ W.^2);
   return logp,G

end



function  logpdiff(U,args,cache)
    M = args.Mat;
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F

    Fxprop = cache.Fxprop
    Dxprop = cache.Dxprop
    res = cache.residual
    Fxprop .= F*U;
    Dxprop .= M*U;

    res .= Fxprop-y

    logp = -0.5/noisesigma^2*dotitself(res);
    logp = logp + sum(log.(scale./((scale^2 .+ Dxprop.^2))));

    return logp

end

function  logpdiffgradi(U,arguments,cache;both=true)
    M = arguments.Mat; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; F = arguments.F

    Fxprop = cache.Fxprop
    Dxprop = cache.Dxprop
    res = cache.residual
    Fxprop .= F*U;
    Dxprop .= M*U;

    res .= Fxprop-y
    logp = -0.5/noisesigma^2*dotitself(res);
    logp = logp + sum(log.(scale./((scale^2 .+ Dxprop.^2))));

    #nt = length(y); N = length(U);
    #for i = 1:nt
            #println(Gr[i,:]')
         #q =  (-((r[i]-y[i])./noisesigma.^2))*Gr[i,:]';
         #G = G + dropdims(q, dims = tuple(findall(size(q) .== 1)...))
     #end
     G= F'*(-((res)./noisesigma.^2))

    Gd =   M'*(-2.0*Dxprop./(scale^2 .+ Dxprop.^2));
    #Gd = Gd'*M; 
    G = G + Gd;

    return logp,G

end

function  logpdiffpartial(ix,U,args,cache,current)
    D = @view args.Mat[:,ix]; 
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; 
    F  = @view args.F[:,ix]

    logp = 0.0

    currx = @view current.xcurr[ix]
    Fxcurr = current.Fxcurr
    Dxcurr =  current.Dxcurr

    xprop = cache.xprop
    xprop[ix] = U

    compx = U-currx
    Fxprop = cache.Fxprop
    Fxprop .= Fxcurr
    Fxprop .+= F*compx

    Dxprop = cache.Dxprop
    Dxprop .= Dxcurr
    Dxprop .+= D*compx

    res = cache.residual
    res .= Fxprop
    res .-=  y

    logp = -0.5/noisesigma^2*dotitself(res);
    logp = logp + sum(log.(scale./((scale^2 .+ Dxprop.^2))));


    return logp

end

function  logpdiffpartialgradi(ix,U,arguments,cache,current;both=true)

    D = @view arguments.Mat[:,ix]; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; F = @view arguments.F[:,ix]

    #(Fxcurr=(F*X0),Dxcurr=(Md1*X0), xcurr = copy(X0))

    currx = @view current.xcurr[ix]
    Fxcurr = current.Fxcurr
    Dxcurr =  current.Dxcurr

    xprop = cache.xprop
    xprop[ix] = U

    compx = U-currx  
    Fxprop = cache.Fxprop
    Fxprop .= Fxcurr
    Fxprop .+= F*compx

    Dxprop = cache.Dxprop
    Dxprop .= Dxcurr
    Dxprop .+= D*compx

    res = cache.residual
    res .= Fxprop
    res .-=  y

    logp = 0.0

    if (both)
        logp = -0.5/noisesigma^2*dotitself(res);
        logp = logp + sum(log.(scale./((scale^2 .+ Dxprop.^2))));
    end

    G = cache.gradiprop

    G[ix] = -F'*res/noisesigma^2

    G[ix] = G[ix] + D'*(-2.0*Dxprop./(scale^2 .+ Dxprop.^2));

    return logp,G[ix]

end

function refreshcurrent!(cache,current;mode=1)
    if (mode ==1)
        current.Mxcurr .= cache.Mxprop
        current.noisecurr .= cache.noiseprop
    elseif (mode==2)
        current.xcurr .= cache.xprop
        current.Fxcurr .= cache.Fxprop
        current.Dxcurr .= cache.Dxprop
    end
end

function repatpartial(lpdf,placeholder,ix,x0,args,cache,current,tuning;mode=1,Np=1,Nruns=1,T=-Inf,extra=copy(x0))::Vector{Float64}
    x = copy(x0[ix])
    dim = length(x)

    logpdf(xp) = lpdf(ix,xp,args,cache,current)
    #logpdf(xp) = lpdf(partialreplace(x0,xp,ix),args,cache)
    density = logpdf(x)

    z = copy(extra[ix])
    u = randn(dim)
    densityz = logpdf(z)

    for i = 1:Nruns
        R = tuning.Cho.L/sqrt(2)
		u .= randn(dim)
		xtilde = x + R * u
		densitytilde = logpdf(xtilde)
		while (log(rand()) >=  density - densitytilde)
			u .= randn(dim)
			xtilde .= x + R * u
			densitytilde = logpdf(xtilde)
		end

		u .= randn(dim)
		xstar = xtilde + R * u
		densitystar = logpdf(xstar)

		while (log(rand()) >=  densitystar - densitytilde)
			u .= randn(dim)
			xstar .= x + R * u
			densitystar = logpdf(xstar)
		end

		u .= randn(dim)
		zstar = xstar + R * u
		densityzstar = logpdf(zstar)

		while (log(rand()) >=  densitystar - densityzstar)
			u .= randn(dim)
			zstar .= xstar + R * u
			densityzstar = logpdf(zstar)
		end

        ratio = densitystar + min(0,density-densityz) - density - min(0,densitystar-densityzstar)
        if (log(rand()) <= ratio)
            x .= xstar
            z .= zstar
			extra[ix] = zstar
			density = densitystar
			densityz = densityzstar
            
        end
        tuning.n = tuning.n + 1

        if (tuning.n <= tuning.Nadapt)
            xmp = copy(tuning.xm)
            tuning.xm .= tuning.xm + (x-tuning.xm)/tuning.n
            xm = tuning.xm
            tuning.C .= ((tuning.n-1)*tuning.C + (x-xm)*(x-xmp)')/tuning.n
            tuning.Cho = cholesky(Hermitian(2.38^2/dim*tuning.C + 1e-11*I(dim)))
        end

        
    end
    logpdf(x)
    refreshcurrent!(cache,current;mode=mode)
    return x
end



function mwgpartial(lpdf,gradilpdf,ix,x00,args,cache,current,tuning;Nruns=5,Np=3,T=5.0,mode=1,extra=nothing)
    x0 = repeat(x00[ix],Np)
    x = copy(x0)
    Nall = length(x)

    xnew = similar(x)
    pnew = similar(xnew)
    p = similar(xnew)

    logpdf_can(xp) = lpdf(ix,xp,args,cache,current)
    #logpdf_can(xp) = lpdf(partialreplace(x00,xp,ix),args,cache)

    logpdf(x) = totallogpdf(x,logpdf_can,Np,T)  

    glpdf_can(xp) = gradilpdf(ix,xp,args,cache,current;both=false)[2]
    #glpdf_can(xp) = gradilpdf(partialreplace(x00,xp,ix),args,cache)[2][ix]

    glpdf(x) = totalloggrad(x,logpdf_can,glpdf_can,Np,T)[2]
    glpdfboth(x) = totalloggrad(x,logpdf_can,glpdf_can,Np,T)

    ll = logpdf(x)   
  
    for _ = 1:Nruns

        tuning.n = tuning.n + 1

        V=randn(Nall)
        xnew .= x+ tuning.Cho.L*V

        lnew = logpdf(xnew)

        U = rand()
        apr = min(1.0,exp(-ll + lnew))
        
        if (U < apr)
            ll = lnew
            x .= xnew
            tuning.acc = tuning.acc +1    
        end


        if (tuning.n <= tuning.Nadapt)
            xmp = copy(tuning.xm)
            tuning.xm .= tuning.xm + (x-tuning.xm)/tuning.n
            xm = tuning.xm
            tuning.C .= ((tuning.n-1)*tuning.C + (x-xm)*(x-xmp)')/tuning.n
            tuning.Cho = cholesky(Hermitian(2.38^2/Nall*tuning.C + 1e-11*I(Nall)))

            # atarg = 0.3
            # V = V/norm(V)
            # z = sqrt(tuning.n^(-0.7) *abs(tuning.acc/tuning.n - atarg)) *  tuning.Cho.L*V
            # #println(z)
            # if (tuning.acc/tuning.n >= atarg)
            #     lowrankupdate!(tuning.Cho, z)
                
            # else
            #     lowrankdowndate!(tuning.Cho, z)
            # end


        end

    end
    
    xret = reweight(x,Np,logpdf_can,T)
    logpdf_can(xret)
    refreshcurrent!(cache,current;mode=mode)

    return xret

end


mutable struct rram    
    n::Int64
    acc::Int64
    Nadapt::Int64
    C::Array{Float64,2}
    Cho::Cholesky{Float64,Array{Float64,2}}
    xm::Vector{Float64}
end


function totalsample(lpdf,glpdf,x0,meas,arg,ix;N=1,Nadapt=1000,Nruns=5,thinning=10,algo=repatpartial,mode=1,Np=2,T=2.0)
    x = copy(x0)
    Npar = length(x0)
    chain = zeros(Npar, Int(floor(N / thinning)))
    Nix = length(ix)   

    if (mode==2)
        cache=(xprop=copy(x0),Fxprop=(arg.F*x0),Dxprop=(arg.Mat*x0), residual=similar(meas),gradiprop=similar(x0))
        current = (Fxcurr=(arg.F*x0),Dxcurr=(arg.Mat*x0), xcurr = copy(x0))   
    else
        cache = []
        current = []
    end

    if algo == repatpartial
        extra = copy(x0)
        tuning = Vector{rram}(undef,Nix)
        for k = 1:Nix
            n = length(ix[k])
            C = 0.1*Array(I(n))
            Cho = cholesky(C)
            xm = copy(x0[ix[k]])
            tuning[k] = rram(1,0,Nadapt,C,Cho,xm)
        end

    elseif (algo==mwgpartial)
        extra = []
        tuning = Vector{rram}(undef,Nix)
        #tuning = Vector{dahmc2}(undef,Nix)
        #tuning = Vector{ahmc}(undef,Nix)
        for k = 1:Nix

            n = length(ix[k])*Np
            C = 5*Array(I(n))
            Cho = cholesky(C)
            xm = repeat(copy(x0[ix[k]]),Np) 
            tuning[k] = rram(1,0,Nadapt,C,Cho,xm)

        end



    pb = ProgressBar(1:N)

    for i in pb
        for k = 1:Nix
            x[ix[k]] = algo(lpdf,glpdf,ix[k],x,arg,cache,current,tuning[k];mode=mode,Nruns=Nruns,extra=extra,Np=Np,T=T)#lpdf,ix,x0,args,cache,current,tuning
            if (i % thinning == 0)
                chain[ix[k], Int(i / thinning)] = x[ix[k]]
            end
           
        end
    end

    return chain,tuning
end

function savechain(c,prefix)
    p = pwd()
    f = p*"/chains/"*prefix*"_"*Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")*".mat"
    matwrite(f,Dict("chain"=>c))

end


N = 200;
x = Vector(range(-0.0,stop=1.0,length=N));    dx = x[2]-x[1];
xi =  copy(x)
measxi = copy(xi[1:3:end])


triangle(x) = 1.0*(x+1)*(x<=0)*(x>=-1) + 1.0*(-x+1)*(x>0)*(x<=1);
heavi(x) = 1.0*(x>=0);
tf(x) = 0+ 0.2*heavi(x-0.1)*heavi(-x+0.15) - 0.5*heavi(x-0.65)*heavi(-x+0.7)+ 0*exp(-60*abs(x-0.2)) - 0*exp(-180*abs(x-0.8)) + 0*triangle(10*(x-0.8)) + 1*heavi(x-0.3)*heavi(-x+0.6);
tf(x) = heavi(x-0.75)*heavi(-x+0.9)+ triangle(10*(x-0.15)) + triangle(10*(x-0.55))*heavi(x-0.55) + 0*triangle(30*(x-0.65))*heavi(-x+0.65) + exp(-70*abs(x-0.4)) # heavi(x-0.45)*heavi(-x+0.55)
sigma = 0.1;  # Standard deviation of the measurement noise.
cw = 500;
kernel(x,y) = sqrt(cw)/sqrt(π)*exp(-cw*(x-y)^2)
tfc(x) = quadgk(y ->  tf(y)*kernel(x,y), 0, 1, rtol=1e-9)[1]
tfc(x) = HCubature.hquadrature(y ->  tf(y)*kernel(x,y), 0.0, 1.0; rtol=1e-6,atol=1e-6,initdiv=7)[1]


F = measurementmatrix(xi,measxi,kernel); # Theory matrix.
gt = tf.(xi);
gtc =  tfc.(measxi); #F*gt
gtc2 = F*gt

Random.seed!(100)

meas = gtc .+ sigma.*randn(size(measxi)); # Simulate the measurements.

lcauchy = 0.015;
acauchy = lcauchy^2; bcauchy = 1;
cauchypriorscale = 0.5;

cd1scale = 0.05;
cd2scale = 0.01;

Md1=difference1(xi);

Md2=difference2(xi);

Mcauchy = smatrix(x,acauchy,bcauchy); MN = size(Mcauchy)[1];

measnoisescaling = 1.0
diff1arg = (F=F, y=meas,Mat=sparse(Md1),scale=cd1scale,noisesigma=measnoisescaling*sigma)
diff2arg = (F=F, y=meas,Mat=sparse(Md2),scale=cd2scale,noisesigma=measnoisescaling*sigma)
varg = (Mat=Mcauchy,F=F,y=meas,scale=cauchypriorscale,noisesigma=measnoisescaling*sigma)

cached=(xprop=similar(xi),Fxprop=(F*xi),Dxprop=(Md1*xi), residual=similar(meas),gradiprop=similar(xi))
cache2d=(xprop=similar(xi),Fxprop=(F*xi),Dxprop=(Md2*xi), residual=similar(meas),gradiprop=similar(xi))
cachev =(xprop = similar(xi), Fxprop = F*xi, Dxprop = Mcauchy*xi,residual=similar(meas),gradiprop=similar(xi))

targetdiff1(w) =-logpdiff(w,diff1arg,cached)
targetdiffgrad1(w) = -logpdiffgradi(w,diff1arg,cached;both=false)[2]#-logpcauchygradi(w,cauchyarg)[2]# 
res1 = Optim.optimize(targetdiff1, targetdiffgrad1,0*randn(N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=false,iterations=1000); inplace = false)
MAP_diff1 = res1.minimizer

targetdiff2(w) =-logpdiff(w,diff2arg,cache2d)
targetdiffgrad2(w) = -logpdiffgradi(w,diff2arg,cached;both=false)[2]#-logpcauchygradi(w,cauchyarg)[2]# 
res2 = Optim.optimize(targetdiff2, targetdiffgrad2,0*randn(N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=false,iterations=30000); inplace = false)
MAP_diff2 = res2.minimizer

targetv(w) = -logpdiff(w,varg,cachev)
targetvgrad(w) = -logpdiffgradi(w,varg,cachev;both=false)[2]#-logpcauchygradi(w,cauchyarg)[2]# 
res2 = Optim.optimize(targetv, targetvgrad,0.00*randn(N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=false,iterations=30000); inplace = false)
MAP_v = res2.minimizer


Ntot = length(xi)

function NUTS_sample(N,arg,cac)
    n_samples, n_adapts = 100, 30
    ss = Vector{Any}(undef,N)
    st = Vector{Any}(undef,N)
    Np = 1
    T = 3.0
    for i = 1:N

        Ntot = length(xi)*Np
        Random.seed!(i)
        initial_θ = repeat(MAP_v,Np)
        global c

        targetdiff(w) = logpdiff(w,arg,cac)
        targetdiffgrad(w) = logpdiffgradi(w,arg,cac;both=true)

        metric = DiagEuclideanMetric(Ntot)# DenseEuclideanMetric(Ntot)# DiagEuclideanMetric(Ntot)
        hamiltonian = Hamiltonian(metric, targetdiff, targetdiffgrad)

        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator =  Leapfrog(initial_ϵ)#  TemperedLeapfrog(initial_ϵ, 1.05)# Leapfrog(initial_ϵ)
        proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,10,1000.0)
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        c, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
        ss[i] = hcat(c...)
        # ss[i] = c
        st[i] = stats
        pr="NUTS_"*string("diff1_NUTS_")
        savechain(c,pr)

    end
    return ss, st
end

#c,b = NUTS_sample(10,diff1arg,cached)


ix = divide(N,1)
algo =  repatpartial#mwgpartial
grfunk = logpdiffpartialgradi#  logpcauchyreppartialgradi# 
funk =  logpdiffpartial#logpcauchyreppartial#logpdiffpartial#
par = diff1arg
mode = 2# 1 
x000  = copy(MAP_diff1) #randn(N,)

cached=(xprop=copy(x000),Fxprop=(F*x000),Dxprop=(Md1*x000), residual=copy(meas),gradiprop=copy(gt))
currentd = (Fxcurr=(F*MAP_diff1),Dxcurr=(Md1*MAP_diff1), xcurr = copy(MAP_diff1))


for t=1:0
    Random.seed!(t)
    xz = copy(MAP_v)#0.1*randn(N,)

    Nsamples = 1500; Nadapt = 1250; thinning = 20
    Np = 1; Nruns = 5; T = 1.0
    
    c,tun=totalsample(funk,grfunk,xz,meas,par,ix;N=Nsamples,Nadapt=Nadapt,thinning=thinning,algo=algo,mode=mode,Np=Np,T=T,Nruns=Nruns)
    savechain(c,"diff1_"*string(nameof(algo))*"_"*string(nameof(funk))*"_N"*string(Nsamples)*"_Nadapt"*string(Nadapt)*"_thin"*string(thinning)*"_Np"*string(Np)*"_Nruns"*string(Nruns)*"_T"*string(T))
    println(effective_sample_size(c[1,:]))
    global c
    global tun
end

