const _FORTRAN_HOTPATHS_AVAILABLE = Ref(false)

function _init_fortran_hotpaths()
    _FORTRAN_HOTPATHS_AVAILABLE[] = isfile(_resolve_fortran_solve_library_path())
    return _FORTRAN_HOTPATHS_AVAILABLE[]
end

function _f_proxycharge2pw_3d!(pwexp, coefs, tab_coefs2pw, nd, porder, npw)
    nd_ref = Ref{Cint}(Cint(nd))
    porder_ref = Ref{Cint}(Cint(porder))
    npw_ref = Ref{Cint}(Cint(npw))

    ccall(
        (:dmk_proxycharge2pw_3d_, _resolve_fortran_solve_library_path()),
        Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{Float64}, Ref{Cint}, Ptr{ComplexF64}, Ptr{ComplexF64}),
        nd_ref,
        porder_ref,
        coefs,
        npw_ref,
        tab_coefs2pw,
        pwexp,
    )

    return pwexp
end

function _f_pw2proxypot_3d!(coefs, pwexp, tab_pw2coefs, nd, porder, npw)
    nd_ref = Ref{Cint}(Cint(nd))
    porder_ref = Ref{Cint}(Cint(porder))
    npw_ref = Ref{Cint}(Cint(npw))

    ccall(
        (:dmk_pw2proxypot_3d_, _resolve_fortran_solve_library_path()),
        Cvoid,
        (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Float64}),
        nd_ref,
        porder_ref,
        npw_ref,
        pwexp,
        tab_pw2coefs,
        coefs,
    )

    return coefs
end

function _f_shiftpw!(pwexp2, pwexp1, wshift, nd, nexp)
    nd_ref = Ref{Cint}(Cint(nd))
    nexp_ref = Ref{Cint}(Cint(nexp))

    ccall(
        (:dmk_shiftpw_, _resolve_fortran_solve_library_path()),
        Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64}),
        nd_ref,
        nexp_ref,
        pwexp1,
        pwexp2,
        wshift,
    )

    return pwexp2
end

function _f_multiply_kernelft!(pwexp, wpwexp, nd, nexp)
    nd_ref = Ref{Cint}(Cint(nd))
    nexp_ref = Ref{Cint}(Cint(nexp))

    ccall(
        (:dmk_multiply_kernelft_, _resolve_fortran_solve_library_path()),
        Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{ComplexF64}, Ptr{Float64}),
        nd_ref,
        nexp_ref,
        pwexp,
        wpwexp,
    )

    return pwexp
end

function _f_density2proxycharge!(fout, fin, umat, ndim, nd, nin, nout)
    ndim_ref = Ref{Cint}(Cint(ndim))
    nd_ref = Ref{Cint}(Cint(nd))
    nin_ref = Ref{Cint}(Cint(nin))
    nout_ref = Ref{Cint}(Cint(nout))
    sc_ref = Ref{Float64}(1.0)

    ccall(
        (:bdmk_density2proxycharge_, _resolve_fortran_solve_library_path()),
        Cvoid,
        (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ptr{Float64}, Ref{Cint}, Ptr{Float64}, Ptr{Float64}, Ref{Float64}),
        ndim_ref,
        nd_ref,
        nin_ref,
        fin,
        nout_ref,
        fout,
        umat,
        sc_ref,
    )

    return fout
end

function _f_proxypot2pot!(fout, fin, umat, ndim, nd, nin, nout)
    ndim_ref = Ref{Cint}(Cint(ndim))
    nd_ref = Ref{Cint}(Cint(nd))
    nin_ref = Ref{Cint}(Cint(nin))
    nout_ref = Ref{Cint}(Cint(nout))

    ccall(
        (:bdmk_proxypot2pot_, _resolve_fortran_solve_library_path()),
        Cvoid,
        (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ptr{Float64}, Ref{Cint}, Ptr{Float64}, Ptr{Float64}),
        ndim_ref,
        nd_ref,
        nin_ref,
        fin,
        nout_ref,
        fout,
        umat,
    )

    return fout
end

function _f_tens_prod_to_potloc!(pot, fvals, ws, tab_loc, ind_loc_cint, ixyz, ndim, nd, n, ntab)
    GC.@preserve pot fvals tab_loc ind_loc_cint ixyz begin
        ccall(
            (:bdmk_tens_prod_to_potloc_, _resolve_fortran_solve_library_path()),
            Cvoid,
            (
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ref{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ref{Cint},
                Ptr{Cdouble},
                Ptr{Cint},
                Ptr{Cint},
            ),
            Cint(ndim),
            Cint(nd),
            Cint(n),
            Cdouble(ws),
            pointer(fvals),
            pointer(pot),
            Cint(ntab),
            pointer(tab_loc),
            pointer(ind_loc_cint),
            pointer(ixyz),
        )
    end

    return pot
end

function _f_tens_prod_trans!(fout, fin, umat_nd, ndim, nin, nout, ifadd)
    GC.@preserve fout fin umat_nd begin
        ccall(
            (:tens_prod_trans_, _resolve_fortran_solve_library_path()),
            Cvoid,
            (
                Ref{Cint},
                Ref{Cint},
                Ptr{Cdouble},
                Ref{Cint},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ref{Cint},
            ),
            Cint(ndim),
            Cint(nin),
            pointer(fin),
            Cint(nout),
            pointer(fout),
            pointer(umat_nd),
            Cint(ifadd),
        )
    end

    return fout
end
