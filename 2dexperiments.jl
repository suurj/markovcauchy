using LinearAlgebra
using Random
using PyPlot
using Interpolations
using SparseArrays
using Optim
using ProgressBars
using StatsBase
using AlphaStableDistributions
using AdvancedHMC
using Dates
using MAT
using MCMCDiagnostics



seed_shift = (@isdefined shift) ? shift : 0

function measurementmatrix(X,MX,kernel)
    s = X[2]-X[1];
    F = zeros(size(MX)[1],size(X)[1])
    for i = 1:size(MX)[1]
        F[i,:] = s.*kernel.(X,MX[i]);
    end
    return F
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

function partialsum!(z, s0, sp, spc)
    N = size(spc)[1]
     for i = 1:N
        r = spc[i]
        z[r] = s0[r] + sp[i]
    end
    return nothing
end

function ravel(q)
    return dropdims(q, dims = tuple(findall(size(q) .== 1)...))
end

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

@inbounds  function measurementmatrix2d(X,Y,Xt,Yt,kernel;constant=10.0)
    dy = abs(Y[2,1]-Y[1,1])
    dx = abs(X[1,2]-X[1,1])
    d = Float64(dy*dx)
    N = length(X);
    Nt = length(Xt)
    #F = zeros(Nt,N)

    minval = 0.0001*d*kernel(0,0,0,0,constant=constant)
    Nth = Threads.nthreads()
    rows = Vector{Vector{Int64}}(undef, Nth)
    cols = Vector{Vector{Int64}}(undef, Nth)
    vals = Vector{Vector{Float64}}(undef, Nth)
    for p = 1:Nth
        rows[p] = []
        cols[p] = []
        vals[p] = []
    end

    Threads.@threads  for i = 1:N
        for j = 1:Nt
            xi = div(i-1,N)+1; 
            yi = rem(i,N)+1
            xj = div(j-1,Nt)+1; 
            yj = rem(j,Nt)+1
            val = d*kernel(X[i],Xt[j],Y[i],Yt[j],constant=constant);
            if (val >= minval)
                #F[j,i] = val
                push!(rows[Threads.threadid()],j)
                push!(cols[Threads.threadid()],i)
                push!(vals[Threads.threadid()],val)
            end
            
            #F[i,j] = d*kernel(X[xi],X[xj],Y[yi],Y[yj],constant=constant);
            #F[j,i] = F[i,j]
        end
    end
    rows = vcat(rows...)
    cols = vcat(cols...)
    vals = vcat(vals...)
    F = sparse(rows,cols,vals,Nt,N)

    return F

end


function  s2dmatrix(x0,a,b)
    N = length(x0);
    dx = x0[2]- x0[1]
   
   M = zeros(N,N);
   for i=2:N-1
       M[i,i-1] = -a/dx^2;
       M[i,i+1] = -a/dx^2;
       M[i,i] = 2*a/dx^2;

   end
   M[1,:] = [2*a/dx^2 -a/dx^2 zeros(1,N-3) -a/dx^2];
   M[N,:] = [-a/dx^2 zeros(1,N-3) -a/dx^2 2*a/dx^2];
   
   M = kron(M,I(N)) + kron(I(N),M)
   M = M + I(size(M)[2])*b;

   return M
end

    function linspace(start,stop,length)
        return Vector(range(start,stop=stop,length=length))
    end


    function partialreplace(orig,replace,ix)
        a = copy(orig)
        a[ix] = replace
        return a
    end

   @inline function meshgrid(x,y) 
        grid_a = [i for i in x, j in y]
        grid_b = [j for i in x, j in y]
    
        return grid_a,grid_b
    end

@inbounds @inline function storevector!(target,source)
    N = length(target)
    @assert N == length(source)
    @simd for i = 1:N
        target[i] = source[i]
    end
end

@inbounds @inline function storesubst!(target,a1,a2)
    N = length(target)
    @assert N == length(a1) == length(a2)
    @simd for i = 1:N
        target[i] = a1[i] - a2[i]
    end
end

@inbounds @inline function storeadd!(target,a1,a2)
    N = length(target)
    @assert N == length(a1) == length(a2)
    @simd for i = 1:N
        target[i] = a1[i] + a2[i]
    end
end


@inbounds @inline function slicemadd!(target,cols,vec,ix)
    N = length(ix)
    @assert N == length(vec)
    #mul!(target,cols[ix[1]],vec[1])
    for i = 1:N
        mul!(target,cols[ix[i]],vec[i],1,1)
    end
end



function  logpdiff_prior(x,args,cache)
    D = args.D;
    noisesigma = args.noisesigma
    scale = args.scale; #y  = args.y; F = args.F

    #Fxprop = cache.Fxprop
    Dprop = cache.Dprop
    #res = cache.residual

    #mul!(Fxprop,F,x)
    mul!(Dprop,D,x)

    #res .= Fxprop-y
    #storesubst!(res,Fxprop,y)
    #logp = -0.5/noisesigma^2*dotitself(res);
    logp = sum(log.(scale./((scale^2 .+ Dprop.^2))));

    return logp

end

function  logpdiffpartial(ixl,nth,x,args,cache,current)
    noisesigma = args.noisesigma
    ix = ixl[nth]
    scale = args.scale; y  = args.y; bscale = args.bscale
    # F =   args.F[:,ix]  
    # Dx =  args.Dx[:,ix]; Dy =  args.Dy[:,ix]
    # Db =  args.Db[:,ix]
    F =   args.Fs[nth] ; D =   args.Ds[nth]
    Ds = args.Ds; Dnz = args.Dnz; 
    Fs = args.Fs; Fnz = args.Fnz

    res = cache.residual
    Dprop = cache.Dprop
    Fxprop = cache.Fxprop
    xprop = cache.xprop

    currx = @view current.xcurr[ix]
    Fxcurr = current.Fxcurr
    Dcurr =  current.Dcurr

    xprop[ix] = x

    # Fnzall = union(vcat(Fnz[ix]...))
    Fnzall = Fnz[nth]
    # Dxnzall = union(vcat(Dxnz[ix]...))
    # Dynzall = union(vcat(Dynz[ix]...))
    Dnzall = Dnz[nth]
    # Dbnzall = union(vcat(Dbnz[ix]...))


    compx = x-currx
    #Fxprop .= Fxcurr
    #Fxprop .+= (F*compx)
    storevector!(view(Fxprop,Fnzall),view(Fxcurr,Fnzall))
    #storevector!(Fxprop,Fxcurr)
    mul!(Fxprop,F,compx,1,1)
    #slicemadd!(Fxprop,Fs,compx,ix)
    
    #Dxprop .= Dxcurr
    #Dxprop .+= Dx*compx
    storevector!(view(Dprop,Dnzall),view(Dcurr,Dnzall))
    #storevector!(Dprop,Dcurr)
    mul!(Dprop,D,compx,1,1)

    #res .= Fxprop
    storesubst!(res,Fxprop,y)
    #storevector!(res,Fxprop)
    #res .-=  y

    resv = @view res[Fnzall]
    Dv = @view Dprop[Dnzall]

    #return   -0.5/noisesigma^2*res'*res -3/2*sum(log.(scale^2 .+ Dxprop.^2 .+ Dyprop.^2)) - sum(log.(bscale^2 .+ Dbprop.^2))
    return   -0.5/noisesigma^2*dotitself(resv) + sum(log.(scale./((scale^2 .+ Dv.^2))));
end

function  logpdiffgradi_prior(x,arguments,cache;both=true)
    D = arguments.D; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; #y  = arguments.y; F = arguments.F

    logp = 0.0

    #Fxprop = cache.Fxprop
    Dprop = cache.Dprop
    G = cache.gradiprop
    #res = cache.residual

    #mul!(Fxprop,F,x)
    mul!(Dprop,D,x)
    #storesubst!(res,Fxprop,y)

    den = (scale^2 .+ Dprop.^2)
    if both
        #logp = -0.5/noisesigma^2*dotitself(res);
        logp =  sum(log.(scale./(den)));
    end

     #mul!(G,F',(-((res)/noisesigma.^2)))
     #G= F'*(-((res)./noisesigma.^2))

    #Gd =   M'*(-2.0*Dprop./den);
    mul!(G,D',(-2.0*Dprop./den))
    #G = G + Gd;

    return logp,G

end

function  logpisodiff_prior(x,args,cache)
    noisesigma = args.noisesigma
    scale = args.scale; #y  = args.y;# F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    #res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    #Fxprop = cache.Fxprop

    #mul!(Fxprop,F,x)
    #res .= Fxprop - y  
    #storesubst!(res,Fxprop,y)
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    return   -sum(log.(bscale^2 .+ Dbprop.^2)) -3/2*sum(log.(scale^2 .+ Dxprop.^2 + Dyprop.^2)) 

end

function  logpisodiffpartial(ixl,nth,x,args,cache,current)
    noisesigma = args.noisesigma
    ix = ixl[nth]
    scale = args.scale; y  = args.y; bscale = args.bscale
    # F =   args.F[:,ix]  
    # Dx =  args.Dx[:,ix]; Dy =  args.Dy[:,ix]
    # Db =  args.Db[:,ix]
    F =   args.Fs[nth]  
    Dx =  args.Dxs[nth]; Dy =  args.Dys[nth]
    Db =  args.Dbs[nth]
    Dxs = args.Dxs; Dxnz = args.Dxnz; 
    Dys = args.Dys; Dynz = args.Dynz
    Dbs = args.Dbs; Dbnz = args.Dbnz
    Dinz = args.Dinz
    Fs = args.Fs; Fnz = args.Fnz

    #res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    xprop = cache.xprop

    currx = @view current.xcurr[ix]
    Fxcurr = current.Fxcurr
    Dxcurr =  current.Dxcurr
    Dycurr = current.Dycurr
    Dbcurr = current.Dbcurr 

    xprop[ix] = x

    # Fnzall = union(vcat(Fnz[ix]...))
    Fnzall = Fnz[nth]
    # Dxnzall = union(vcat(Dxnz[ix]...))
    Dxnzall = Dxnz[nth]
    # Dynzall = union(vcat(Dynz[ix]...))
    Dynzall = Dynz[nth]
    # Dbnzall = union(vcat(Dbnz[ix]...))
    Dbnzall = Dbnz[nth]
    Diall = Dinz[nth]
    #Diall = union(Dxnzall,Dynzall)

    compx = x-currx
    #Fxprop .= Fxcurr
    #Fxprop .+= (F*compx)
    storevector!(view(Fxprop,Fnzall),view(Fxcurr,Fnzall))
    #storevector!(Fxprop,Fxcurr)
    mul!(Fxprop,F,compx,1,1)
    #slicemadd!(Fxprop,Fs,compx,ix)
    
    #Dxprop .= Dxcurr
    #Dxprop .+= Dx*compx
    storevector!(view(Dxprop,Dxnzall),view(Dxcurr,Dxnzall))
    #storevector!(Dxprop,Dxcurr)
    mul!(Dxprop,Dx,compx,1,1)

    #Dyprop .= Dycurr
    #Dyprop .+= Dy*compx
    storevector!(view(Dyprop,Dynzall),view(Dycurr,Dynzall))
    #storevector!(Dyprop,Dycurr)
    mul!(Dyprop,Dy,compx,1,1)

    #Dbprop .= Dbcurr
    #Dbprop .+= Db*compx
    storevector!(view(Dbprop,Dbnzall),view(Dbcurr,Dbnzall))
    #storevector!(Dbprop,Dbcurr)
    mul!(Dbprop,Db,compx,1,1)

    #res .= Fxprop
    storesubst!(res,Fxprop,y)
    #storevector!(res,Fxprop)
    #res .-=  y

    resv = @view res[Fnzall]
    Dxv = @view Dxprop[Diall]
    Dyv = @view Dyprop[Diall]
    Dbv = @view Dbprop[Dbnzall]

    #return   -0.5/noisesigma^2*res'*res -3/2*sum(log.(scale^2 .+ Dxprop.^2 .+ Dyprop.^2)) - sum(log.(bscale^2 .+ Dbprop.^2))
    return   -0.5/noisesigma^2*dotitself(resv) - sum(log.(bscale^2 .+ Dbv.^2)) -3/2*sum(log.(scale^2 .+ Dxv.^2 + Dyv.^2)) 

end

function logpisodiffgradi_prior(x,args,cache;both=true)
    noisesigma = args.noisesigma
    scale = args.scale; #y  = args.y; #F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    logp = 0.0

    #res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    #Fxprop = cache.Fxprop
    G = cache.gradiprop

    #Fxprop .= F*x
    #mul!(Fxprop,F,x)
    #storesubst!(res,Fxprop,y)
    #res .= Fxprop - y  
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    #G .= F'*(-((res)./noisesigma.^2))
    #mul!(G,F',-((res)/noisesigma.^2))

    den = (scale^2 .+ Dyprop.^2 + Dxprop.^2)

    if both
        logp =  -3/2*sum(log.(den)) - sum(log.(bscale^2 .+ Dbprop.^2))
    end

    #Gd1 =  Dx'*(-3.0*Lxx./(scale^2 .+ Lxx.^2));
    #G .= G  -3.0*(Dx)'*(Dxprop./(scale^2 .+ Dyprop.^2 + Dxprop.^2))  - 3.0*(Dy)'*(Dyprop./(scale^2 .+ Dyprop.^2 + Dxprop.^2))   -Db'*(2.0*Dbprop./(bscale^2 .+ Dbprop.^2));
    
    mul!(G,-3.0*(Dx)',(Dxprop./den))
    mul!(G,-3.0*(Dy)',(Dyprop./den),1,1)
    mul!(G,-Db',(2.0*Dbprop./(bscale^2 .+ Dbprop.^2)),1,1)

    return logp, G 
end



function regmatrices_first(dim)
    reg1d = spdiagm(Pair(0,-1*ones(dim))) + spdiagm(Pair(1,ones(dim-1))) + spdiagm(Pair(-dim+1,ones(1))) ;reg1d[dim,dim] = 0
    #reg1d = reg1d[1:dim-1,:]
    iden = I(dim)
    regx = kron(reg1d,iden)
    regy = kron(iden,reg1d)

    rmxix = sum(abs.(regx) ,dims=2) .< 2
    rmyix = sum(abs.(regy) ,dims=2) .< 2
    boundary = ((rmxix + rmyix)[:]) .!= 0
    q = findall(boundary .== 1)
    regx = regx[setdiff(1:dim^2,q), :] 
    regy = regy[setdiff(1:dim^2,q), :] 
    
    s = length(q)
    bmatrix = sparse(zeros(s,dim*dim))
    for i=1:s
        v = q[i]
        bmatrix[i,v] = 1
    end
    #bmatrix = bmatrix[i,i] .= 1

    return regx,regy,bmatrix
end

function regmatrices_second(dim)
    reg1d = spdiagm(Pair(0,2*ones(dim))) + spdiagm(Pair(1,-1*ones(dim-1))) + spdiagm(Pair(-1,-1*ones(dim-1))) #;reg1d[dim,dim] = 0
    #reg1d = reg1d[1:dim-1,:]
    iden = I(dim)
    regx = kron(reg1d,iden)
    regy = kron(iden,reg1d)

    rmxix = sum(abs.(regx), dims=2) .< 4
    rmyix = sum(abs.(regy), dims=2) .< 4
    boundary = ((rmxix + rmyix)[:]) .!= 0
    q = findall(boundary .== 1)
    regx = regx[setdiff(1:dim^2,q), :] 
    regy = regy[setdiff(1:dim^2,q), :] 

    #q = findall(boundary .== 1)
    #bmatrix = zeros(dim*dim,dim*dim)
    #for i in q
    #     bmatrix[i,i] = 1
    #end

    h2 = sparse(zeros(2,N))
    h2[1,1] = 1; h2[2,N] = 1
    h1 = hcat([sparse(zeros(dim-2)),I(dim-2),sparse(zeros(dim-2))]...)
    bmatrix = [ kron(h2,I(dim)); kron(h1,h2) ]

    return regx,regy,bmatrix
end

function spdematrix(xs,a,b)
    dx = abs(xs[2]-xs[1])
    N = length(xs)
    
    M = sparse(zeros(N,N))
    for i=2:N-1
        M[i,i-1] = -a/dx^2;
        M[i,i+1] = -a/dx^2;
        M[i,i] = 2*a/dx^2;# +b;

    end

    M[1,1] = 1 # Option 1
    M[N,N] = 1
    M[N,N-1] = -1 

    #M[1,:] = [2*a/dx^2;-a/dx^2; zeros(N-3);-a/dx^2]; # Option 2
    #M[N,:] = [-a/dx^2; zeros(N-3); -a/dx^2;2*a/dx^2];
    
    M = kron(M,I(N)) + kron(I(N),M)
    M = M + I(size(M,2))*b;

    return M
end       

function nzrows(M::SparseVector)
    return unique(M.nzind)
end

function nzrows(M::SparseMatrixCSC)
    return unique(M.rowval)
end

function splitmatrix(M::SparseMatrixCSC,ixlist::Array{Array{Int64,1},1})
    N = length(ixlist)
    zl = Vector{Vector{Int64}}(undef,N)
    Ml = Vector{AbstractSparseArray}(undef,N)
    for i =1:N
        Mi = M[:,ixlist[i]]
        nz = nzrows(Mi)
        zl[i] = nz
        Ml[i] = Mi
        #push!(zl,nz)
        #push!(Ml,Mi)
    end
    return Ml,zl
end

function splitmatrix(M::SparseMatrixCSC)
    N = size(M)[2]
    zl = Vector{Vector{Int64}}(undef,N)
    Ml = Vector{AbstractSparseArray}(undef,N)
    for i =1:N
        Mi = M[:,i]
        nz = nzrows(Mi)
        zl[i] = nz
        Ml[i] = Mi
        #push!(zl,nz)
        #push!(Ml,Mi)
    end
    return Ml,zl
end

mutable struct rram    
    n::Int64
    acc::Int64
    Nadapt::Int64
    C::Array{Float64,2}
    Cho::Cholesky{Float64,Array{Float64,2}}
    xm::Vector{Float64}
end




function refreshcurrent!(cache,current;mode=1,arg=nothing,ixl=nothing,nth=nothing)
    if isnothing(arg)
        if (mode ==1)
            current.Mxcurr .= cache.Mxprop
            current.noisecurr .= cache.noiseprop
        elseif (mode==2)
            # current.xcurr .= cache.xprop
            # current.Fxcurr .= cache.Fxprop
            # current.Dcurr .= cache.Dprop
            storevector!(current.xcurr, cache.xprop)
            storevector!(current.Fxcurr, cache.Fxprop)
            storevector!(current.Dcurr, cache.Dprop)

        elseif (mode==3)
            # current.xcurr .= cache.xprop
            # current.Fxcurr .= cache.Fxprop
            # current.Dxcurr .= cache.Dxprop
            # current.Dycurr .= cache.Dyprop
            # current.Dbcurr .= cache.Dbprop
            storevector!(current.xcurr, cache.xprop)
            storevector!(current.Fxcurr, cache.Fxprop)
            storevector!(current.Dxcurr, cache.Dxprop)
            storevector!(current.Dycurr, cache.Dyprop)
            storevector!(current.Dbcurr, cache.Dbprop)
        else
            error("")
        end

    else
        if (mode ==1)
            current.Mxcurr .= cache.Mxprop
            current.noisecurr .= cache.noiseprop
        elseif (mode==2)
            storevector!(view(current.xcurr,ixl[nth]), view(cache.xprop,ixl[nth]))
            storevector!(view(current.Fxcurr,arg.Fnz[nth]), view(cache.Fxprop,arg.Fnz[nth]))
            storevector!(view(current.Dcurr,arg.Dnz[nth]), view(cache.Dprop,arg.Dnz[nth]))
            #current.xcurr .= cache.xprop
            #current.Fxcurr .= cache.Fxprop
            #current.Dcurr .= cache.Dprop

        elseif (mode==3)
            # current.xcurr .= cache.xprop
            # current.Fxcurr .= cache.Fxprop
            # current.Dxcurr .= cache.Dxprop
            # current.Dycurr .= cache.Dyprop
            # current.Dbcurr .= cache.Dbprop
            storevector!(view(current.xcurr,ixl[nth]), view(cache.xprop,ixl[nth]))
            storevector!(view(current.Fxcurr,arg.Fnz[nth]), view(cache.Fxprop,arg.Fnz[nth]))
            storevector!(view(current.Dxcurr,arg.Dxnz[nth]), view(cache.Dxprop,arg.Dxnz[nth]))
            storevector!(view(current.Dycurr,arg.Dynz[nth]), view(cache.Dyprop,arg.Dynz[nth]))
            storevector!(view(current.Dbcurr,arg.Dbnz[nth]), view(cache.Dbprop,arg.Dbnz[nth]))
  
        else
            error("")
        end
    end
end

function mwgpartial(lpdf,gradilpdf,ixl,nth,x00,args,cache,current,tuning;Nruns=5,Np=3,T=5.0,mode=1,extra=nothing)
    ix = ixl[nth]
    x0 = repeat(x00[ix],Np)
    x = x0# copy(x0)
    Nall = length(x)

    xnew = similar(x)

    logpdf_can(xp) = lpdf(ixl,nth,xp,args,cache,current)
    #logpdf_can(xp) = lpdf(partialreplace(x00,xp,ix),args,cache)

    logpdf(x) = logpdf_can(x)  

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
    
    xret =  x#reweight(x,Np,logpdf_can,T)
    logpdf_can(xret)
    refreshcurrent!(cache,current;mode=mode,arg=args,ixl=ixl,nth=nth)

    return xret

end

function repatpartial(lpdf,placeholder,ixl,nth,x0,args,cache,current,tuning;mode=1,Np=1,Nruns=1,T=-Inf,extra=copy(x0))::Vector{Float64}
    ix = ixl[nth]
    x = copy(x0[ix])
    dim = length(x)

    logpdf(xp) = lpdf(ixl,nth,xp,args,cache,current)
    #logpdf(xp) = lpdf(partialreplace(x0,xp,ixl[nth]),args,cache)
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
    refreshcurrent!(cache,current;mode=mode,arg=args,nth=nth,ixl=ixl)
    return x
end

function totalsample(lpdf,glpdf,x0,meas,arg,ixl;N=1,Nadapt=1000,Nruns=5,thinning=10,algo=repatpartial,mode=2,Np=2,T=2.0)
    x = copy(x0)
    Npar = length(x0)
    chain = zeros(Npar, Int(floor(N / thinning)))
    Nix = length(ixl)   


    if (mode==2)
        cache=(xprop=copy(x0),Fxprop=(arg.F*x0),Dprop=(arg.D*x0), residual=similar(arg.y),gradiprop=similar(x0))
        current = (Fxcurr=(arg.F*x0),Dcurr=(arg.D*x0), xcurr = copy(x0))  
    elseif (mode==3)
        cache =(xprop=copy(x0),Fxprop=(arg.F*x0),Dxprop=(arg.Dx*x0),Dyprop=(arg.Dy*x0), Dbprop=(arg.Db*x0),  residual=arg.F*x0-arg.y,gradiprop=similar(x0))   
        current = (xcurr = copy(x0), Dxcurr = arg.Dx*x0, Dycurr = arg.Dy*x0, Dbcurr = arg.Db*x0, Fxcurr = F*x0)
    else
        error("")
    end

    if algo == repatpartial
        extra = copy(x0)
        tuning = Vector{rram}(undef,Nix)
        for k = 1:Nix
            n = length(ixl[k])
            C = 0.1*Array(I(n))
            Cho = cholesky(C)
            xm = copy(x0[ixl[k]])
            tuning[k] = rram(1,0,Nadapt,C,Cho,xm)
        end

    elseif (algo==mwgpartial)
        extra = []
        tuning = Vector{rram}(undef,Nix)
        #tuning = Vector{dahmc2}(undef,Nix)
        #tuning = Vector{ahmc}(undef,Nix)
        for k = 1:Nix

            n = length(ixl[k])*Np
            C = 0.1*Array(I(n))
            Cho = cholesky(C)
            xm = repeat(copy(x0[ixl[k]]),Np) 
            tuning[k] = rram(1,0,Nadapt,C,Cho,xm)


        end
    end

    pb = ProgressBar(1:N)

    for i in pb
        for k = 1:Nix
            x[ixl[k]] = algo(lpdf,glpdf,ixl,k,x,arg,cache,current,tuning[k];mode=mode,Nruns=Nruns,extra=extra,Np=Np,T=T)#lpdf,ix,x0,args,cache,current,tuning
            if (i % thinning == 0)
                chain[ixl[k], Int(i / thinning)] = x[ixl[k]]
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

cw = 150.0
kernel(xi,xj,yi,yj;constant=cw) = constant/pi*exp(-constant*((xi-xj)^2 + (yi-yj)^2) )
tf(x,y) =  1.0*exp.(-20*sqrt.((x .- 0.3).^2 .+ (y .- 0.3).^2)) + 1*((y-x) .< 0.7).*((y-x) .>= -0.7).*((-y-x) .<= 0.8).*((-y-x) .>= 0.4).*(((-x+y) .+ 0.7)/1.39)  + 1*(-x+y .< 1).*(-x+y .>= 0.8).*(abs.(x) .<= 1).*(abs.(y) .<= 1) + ( 4*0.25 .- 4*sqrt.((x.-0.5).^2+(y.+0.6).^2)).*(sqrt.((x.-0.5).^2+(y.+0.6).^2) .<= 0.25);

Random.seed!(1)

noisevar = 0.01^2

Nbig = 300
dimbig = Nbig
N = 180
dim = N
Nmeas = 100
dimmeas = Nmeas

xsbig = -1+1/dimbig:2/dimbig:1-1/(dimbig)
ysbig = -1+1/dimbig:2/dimbig:1-1/(dimbig)

xs = -1+1/dim:2/dim:1-1/(dim)
ys = -1+1/dim:2/dim:1-1/(dim)

xibig = linspace(-1,1,dimbig)
Ybig,Xbig = meshgrid(xibig,xibig)

ximeas = linspace(-1,1,dimmeas)
Ymeas,Xmeas = meshgrid(ximeas,ximeas)
Y,X = meshgrid(xs,xs)


Zbig = tf(Xbig,Ybig)
@time Fbig = measurementmatrix2d(Xbig,Ybig,Xmeas,Ymeas,kernel,constant=cw); # Theory matrix for measurements.
@time F = measurementmatrix2d(X,Y,Xmeas,Ymeas,kernel,constant=cw); # Theory matrix for inversion. 



zx = Fbig*Zbig[:]
mebig = reshape(zx,(dimmeas,dimmeas))
y = mebig[:] + randn(Nmeas*Nmeas,)*sqrt(noisevar)


xp = randn(N*N)
ixl = divide(N*N,1)

Dx,Dy,Db = regmatrices_first(N);
Dxs, Dxnz = splitmatrix(Dx,ixl); Dys, Dynz = splitmatrix(Dy,ixl); Dbs, Dbnz = splitmatrix(Db,ixl); #Fs, Fnz = splitmatrix(F,ixl)
S = spdematrix(xs,0.001,0.1)
D = [Dx;Dy;Db]
Ds, Dnz = splitmatrix(D,ixl)
Dinz = [union(Dxnz[i],Dynz[i]) for i = 1:N^2]
argi  = (Dinz = Dinz, Dxs = Dxs, Dxnz = Dxnz , Dys = Dys, Dynz = Dynz, Dbs = Dbs, Dbnz = Dbnz, Dx = Dx, Dy = Dy, D=D,noisesigma = sqrt(noisevar), scale = 0.03, bscale = 0.03, Db = Db, Ds = Ds, Dnz = Dnz )
argis  = ( D=S, F = F,noisesigma = sqrt(noisevar), scale = 0.3, y = y, bscale = 1.0 )

cacheiso =(xprop=zeros(N*N),Dxprop=zeros(size(Dx)[1]),Dyprop=zeros(size(Dy)[1]),Dbprop=zeros(size(Db)[1]),gradiprop=zeros(N*N))
cached =(xprop=zeros(N*N),Dprop=zeros(size(D)[1]),gradiprop=zeros(N*N))
caches =(xprop=zeros(N*N),Fxprop=zeros(Nmeas^2),Dprop=zeros(size(S)[1]), residual=similar(y),gradiprop=zeros(N*N))

currentiso = (xcurr = copy(xp), Dxcurr = Dx*xp, Dycurr = Dy*xp, Dbcurr = Db*xp, Fxcurr = F*xp)
currentaniso = (xcurr = copy(xp), Dcurr = D*xp, Fxcurr = F*xp)



target1iso(x) = -logpisodiff(x,argi,cacheiso)# -logpdiff(w,diff1arg,cached)## 
target1isograd(x) = -logpisodiffgradi(x,argi,cacheiso;both=false)[2]#  -logpdiffgradi(w,diff1arg,cached;both=false)[2]#
res = Optim.optimize(target1iso, target1isograd, 0*randn(N*N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=true,iterations=700); inplace = false)
MAP_diff1iso = res.minimizer
imshow(reshape(MAP_diff1iso,(N,N))); colorbar(); clim(-0.1,1.1)

target1(x) = -logpdiff(x,argi,cached)# -logpdiff(w,diff1arg,cached)## 
target1grad(x) = -logpdiffgradi(x,argi,cached;both=false)[2]#  -logpdiffgradi(w,diff1arg,cached;both=false)[2]#
res = Optim.optimize(target1, target1grad, 0*randn(N*N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=true,iterations=700); inplace = false)
MAP_diff1 = res.minimizer
imshow(reshape(MAP_diff1,(N,N))); colorbar(); clim(-0.1,1.2)

targets(x) = -logpdiff(x,argis,caches)# -logpdiff(w,diff1arg,cached)## 
targetsgrad(x) = -logpdiffgradi(x,argis,caches;both=false)[2]#  -logpdiffgradi(w,diff1arg,cached;both=false)[2]#
res = Optim.optimize(targets, targetsgrad, 0*randn(N*N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=true,iterations=700); inplace = false)
MAP_s = res.minimizer
imshow(reshape(MAP_s,(N,N))); colorbar(); clim(-0.5,12)



function NUTS_sample(Niter,arg,cac)
    n_samples, n_adapts = 2000, 1000
    ss = Vector{Any}(undef,Niter)
    st = Vector{Any}(undef,Niter)
    Np = 1
    T = 1
    for i = (1:Niter) .+ seed_shift

        Ntot = N^2
        Random.seed!(i)
        println(i)
        initial_θ = copy(MAP_diff1)
        global c

        targetdiff(w) = -logpdiff(x,arg,cac)
        targetdiffgrad(w) = logpdiffgradi(w,arg,cac;both=true) 

        metric = DiagEuclideanMetric(Ntot)# DenseEuclideanMetric(Ntot)# DiagEuclideanMetric(Ntot)
        hamiltonian = Hamiltonian(metric, targetdiff, targetdiffgrad)

        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator =  Leapfrog(initial_ϵ)#  TemperedLeapfrog(initial_ϵ, 1.05)# Leapfrog(initial_ϵ)
        proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,8,1000.0)
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        c, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
        #ss[i] = hcat(c...)
        c = hcat(c...)
        # ss[i] = c
        #st[i] = stats
        pr="NUTS_"*string("anisodiff1d_prior")
        savechain(c,pr)

    end
    return ss, st
end
ch,b = NUTS_sample(1,argi,cached)


algo =  repatpartial#mwgpartial#
grfunk = nothing#  
funk =  logpdiffpartial
par = argi
mode = 2# 1 
#x000  = randn(N*N,)


for t=1:0
    Random.seed!(t)
    xz = copy(MAP_diff1) 

    Nsamples = 100; Nadapt = 10; thinning = 10
    Np = 1; Nruns = 1; T = 1.0
    
    c,tun=totalsample(funk,grfunk,xz,y,par,ixl;N=Nsamples,Nadapt=Nadapt,thinning=thinning,algo=algo,mode=mode,Np=Np,T=T,Nruns=Nruns)
    savechain(c,"anisodiff1d_normal_"*string(nameof(algo))*"_"*string(nameof(funk))*"_N"*string(Nsamples)*"_Nadapt"*string(Nadapt)*"_thin"*string(thinning)*"_Np"*string(Np)*"_Nruns"*string(Nruns)*"_T"*string(T))
    println(effective_sample_size(c[1,:]))
    global c
    global tun
end
