"""
pfile_imas.jl

Read Osborne p-files and populate an IMAS data dictionary
with core profile data.  The workflow mirrors the gfile loader:

    gfile  = EFIT.readg("g190904.01906_324")
    EFIT.geqdsk2imas!(gfile, dd.equilibrium.time_slice[])

    pfile  = read_pfile("p190904.01906_324")
    pfile2imas!(pfile, dd; time_index=1)

This script is designed to be as faithful as possible to the original OMFITpFile.to_omas() implementation in OMFIT

Prerequisites
─────────────
  • dd.equilibrium must already be populated (e.g. from a g-file) so that
    the psinorm → rho_tor_norm coordinate mapping can be performed.
  • Tested against FUSE / IMAS.jl conventions (COCOS 11 internally in FUSE;
    pfiles are COCOS 1 – CCW-φ, CW-θ – so sign_Ip factors are applied where
    the Python OMFIT code applies them).

Outputs
───────
  • dd.core_profiles.profiles_1d[time_index] fully populated.
  • The raw PFile struct, which contains per-quantity DataFrames suitable
    for direct plotting / further analysis.
"""

module PFileIMAS

using DataFrames
using Interpolations
using Statistics: mean
using IMAS

export PFile, PFileQuantity, read_pfile, pfile2imas!, pfile_to_dataframe

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

"""
One quantity stored in a pfile (e.g. `ne`, `te`, `omgeb`, …).
"""
struct PFileQuantity
    psinorm    :: Vector{Float64}
    data       :: Vector{Float64}
    derivative :: Vector{Float64}
    units      :: String
    description:: String
end

"""
Complete parsed pfile.

`quantities` is a Dict{String,PFileQuantity} keyed by the pfile variable name.
`ion_NZA` holds the ion-species table as a DataFrame with columns :N, :Z, :A.
Following OMFIT convention the last two rows are [main_ion, beam_ion] and
earlier rows are impurities.
"""
struct PFile
    quantities :: Dict{String,PFileQuantity}
    ion_NZA    :: Union{DataFrame,Nothing}
end

# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

const DESCRIPTIONS = Dict(
    "ne"    => "Electron density",
    "te"    => "Electron temperature",
    "ni"    => "Ion density",
    "ti"    => "Ion temperature",
    "nb"    => "Fast ion density",
    "pb"    => "Fast ion pressure",
    "ptot"  => "Total pressure",
    "omeg"  => "Toroidal rotation: VTOR/R",
    "omegp" => "Poloidal rotation: Bt*VPOL/(RBp)",
    "omgvb" => "VxB rotation term in ExB rotation frequency",
    "omgpp" => "Diamagnetic term in ExB rotation frequency",
    "omgeb" => "ExB rotation frequency: OMGPP+OMGVB = Er/(RBp)",
    "er"    => "Radial electric field from force balance: OMGEB*RBp",
    "ommvb" => "Main ion VXB term of Er/RBp",
    "ommpp" => "Main ion pressure term of Er/RBp",
    "omevb" => "Electron VXB term of Er/RBp",
    "omepp" => "Electron pressure term of Er/RBp",
    "kpol"  => "KPOL=VPOL/Bp",
    "omghb" => "Hahm-Burrell ExB velocity shearing rate",
)

"""
    read_pfile(filename) -> PFile

Parse an Osborne p-file and return a PFile struct.
"""
function read_pfile(filename::AbstractString) :: PFile
    quantities = Dict{String,PFileQuantity}()
    ion_NZA    = nothing

    lines = readlines(filename)
    filter!(!isempty ∘ strip, lines)

    i = 1
    while i <= length(lines)
        header = strip(lines[i])
        isempty(header) && (i += 1; continue)

        m_npts = match(r"^(\d+)\s+", header)
        m_npts === nothing && (i += 1; continue)
        npts = parse(Int, m_npts.captures[1])

        # ── N Z A ion species block ──────────────────────────────────────────
        if occursin("N Z A of ION SPECIES", header)
            rows = []
            for j in i+1 : i+npts
                j > length(lines) && break
                vals = parse.(Float64, split(strip(lines[j])))
                push!(rows, (N=vals[1], Z=vals[2], A=vals[3]))
            end
            ion_NZA = DataFrame(rows)
            i += npts + 1
            continue
        end

        # ── normal quantity block ────────────────────────────────────────────
        # header: "256 psinorm ne(10^20/m^3) dne/dpsiN"
        m_key = match(r"^\d+\s+\S+\s+(\w+)\(([^)]*)\)", header)
        if m_key === nothing
            m_key = match(r"^\d+\s+\S+\s+(\S+)", header)
            key   = m_key !== nothing ? String(m_key.captures[1]) : "unknown_$(i)"
            units = ""
        else
            key   = String(m_key.captures[1])
            units = String(m_key.captures[2])
        end

        psinorm    = Vector{Float64}(undef, npts)
        data       = Vector{Float64}(undef, npts)
        derivative = Vector{Float64}(undef, npts)

        for j in 1:npts
            idx = i + j
            idx > length(lines) && break
            vals = parse.(Float64, split(strip(lines[idx])))
            psinorm[j]    = vals[1]
            data[j]       = length(vals) >= 2 ? vals[2] : 0.0
            derivative[j] = length(vals) >= 3 ? vals[3] : 0.0
        end

        desc = get(DESCRIPTIONS, key, key)
        quantities[key] = PFileQuantity(psinorm, data, derivative, units, desc)
        i += npts + 1
    end

    return PFile(quantities, ion_NZA)
end

# ──────────────────────────────────────────────────────────────────────────────
# Convenience: pfile → DataFrame (for plotting / inspection)
# ──────────────────────────────────────────────────────────────────────────────

"""
    pfile_to_dataframe(pfile) -> Dict{String, DataFrame}

Return a dictionary of DataFrames, one per quantity, each with columns
:psinorm, :data, :derivative, :units, :description.
"""
function pfile_to_dataframe(pfile::PFile) :: Dict{String,DataFrame}
    out = Dict{String,DataFrame}()
    for (key, q) in pfile.quantities
        out[key] = DataFrame(
            psinorm     = q.psinorm,
            data        = q.data,
            derivative  = q.derivative,
            units       = fill(q.units, length(q.psinorm)),
            description = fill(q.description, length(q.psinorm)),
        )
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate mapping helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    _dedup_monotone(x, y) -> (x_clean, y_clean)

Remove duplicate x values (keeping first occurrence) and ensure strict
monotonic increase.  Pfiles often have several repeated psinorm=1.0 points
at the edge which cause Interpolations.jl to error.
"""
function _dedup_monotone(x::Vector{Float64}, y::Vector{Float64})
    idx = Int[]
    seen = Set{Float64}()
    for (i, xi) in enumerate(x)
        if xi ∉ seen
            push!(idx, i)
            push!(seen, xi)
        end
    end
    # also ensure strictly increasing after dedup
    keep = [idx[1]]
    for j in 2:length(idx)
        if x[idx[j]] > x[keep[end]]
            push!(keep, idx[j])
        end
    end
    return x[keep], y[keep]
end

"""
    _build_psin_to_rhotn(dd, time_index) -> (psin_eq, rhotn_eq)

Extract the (psinorm, rho_tor_norm) mapping from the equilibrium already
loaded in dd.
"""
function _build_psin_to_rhotn(dd, time_index::Int)
    ts     = dd.equilibrium.time_slice[time_index]
    prof1d = ts.profiles_1d

    psi    = prof1d.psi
    rhotn  = prof1d.rho_tor_norm
    psi_ax = ts.global_quantities.psi_axis
    psi_b  = ts.global_quantities.psi_boundary

    psin_eq = (psi .- psi_ax) ./ (psi_b - psi_ax)

    # deduplicate equilibrium grid too just in case
    psin_eq, rhotn = _dedup_monotone(psin_eq, rhotn)
    return psin_eq, rhotn
end

"""
    _interp_to_rhotn(psin_q, data_q, psin_eq, rhotn_eq, rhotn_out)

Interpolate a per-quantity psinorm grid onto the target rhotn_out grid.
Handles duplicate psinorm points (common at the edge of pfiles).
If the quantity has only 1 point (degenerate), returns a constant array.
"""
function _interp_to_rhotn(
        psin_q    :: Vector{Float64},
        data_q    :: Vector{Float64},
        psin_eq   :: Vector{Float64},
        rhotn_eq  :: Vector{Float64},
        rhotn_out :: Vector{Float64},
    ) :: Vector{Float64}

    # guard: length-1 arrays can't be interpolated — return constant
    if length(psin_q) <= 1
        val = length(data_q) == 1 ? data_q[1] : 0.0
        return fill(val, length(rhotn_out))
    end

    # 1. deduplicate the quantity's own psinorm grid
    psin_q_clean, data_q_clean = _dedup_monotone(psin_q, data_q)

    # guard: dedup may reduce to 1 point
    if length(psin_q_clean) <= 1
        val = length(data_q_clean) == 1 ? data_q_clean[1] : 0.0
        return fill(val, length(rhotn_out))
    end

    # 2. map this quantity's psinorm → rhotn via the equilibrium mapping
    itp_psin2rho = linear_interpolation(psin_eq, rhotn_eq; extrapolation_bc=Line())
    rhotn_q = itp_psin2rho.(clamp.(psin_q_clean, psin_eq[1], psin_eq[end]))

    # 3. deduplicate rhotn_q (mapping can collapse near-identical psin values)
    rhotn_q_clean, data_q_clean2 = _dedup_monotone(rhotn_q, data_q_clean)

    # guard: dedup may reduce to 1 point
    if length(rhotn_q_clean) <= 1
        val = length(data_q_clean2) == 1 ? data_q_clean2[1] : 0.0
        return fill(val, length(rhotn_out))
    end

    # 4. interpolate data onto the output rhotn grid
    itp_data = linear_interpolation(rhotn_q_clean, data_q_clean2; extrapolation_bc=Line())
    return itp_data.(clamp.(rhotn_out, rhotn_q_clean[1], rhotn_q_clean[end]))
end

# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

"""
    pfile2imas!(pfile, dd; time_index=1)

Populate dd.core_profiles.profiles_1d[time_index] from a parsed PFile.

The equilibrium at dd.equilibrium.time_slice[time_index] must already be
populated (e.g. loaded from a g-file) because it is used to convert the
pfile's per-quantity psinorm grids to rho_tor_norm.

If dd.core_profiles.profiles_1d[time_index].grid.rho_tor_norm is already
populated (e.g. from a ZIPFIT/FUSE init), that grid is preserved and all
pfile quantities are interpolated onto it. This prevents grid size mismatches
when overlaying pfile data onto an existing dd.

If no grid exists yet, one is built from ne's psinorm grid via the equilibrium
psinorm→rhotn mapping.

Sign conventions follow the Python OMFIT OMFITpFile.to_omas() exactly:
  • pfiles use COCOS 1 (CCW-φ, CW-θ)
  • sign_Ip is derived from the equilibrium Ip and applied to rotation
    quantities where OMFIT applies it
  • Fast ion pressure pb: isotropic assumption →
      pressure_fast_perpendicular = (2/3) pb
      pressure_fast_parallel      = (1/3) pb

Unit conversions (same as OMFIT):
  ne, ni, nb, nzN  [10^20/m^3]  → [m^-3]    × 1e20
  te, ti           [keV]        → [eV]       × 1e3
  pb, ptot         [kPa]        → [Pa]       × 1e3
  omgeb            [krad/s]     → [rad/s]    × 1e3
  er               [kV/m]       → [V/m]      × 1e3
  vtor, vpol       [km/s]       → [m/s]      × 1e3

Notes on what is NOT mapped:
  • electron rotation fields (omevb, omepp) — not present in this IMASdd version
  • ion rotation sub-fields — stored as velocity.toroidal / velocity.poloidal
    which are standard IMAS fields
"""
function pfile2imas!(pfile::PFile, dd; time_index::Int=1)

    qs = pfile.quantities

    # ── coordinate mapping from equilibrium ─────────────────────────────────
    psin_eq, rhotn_eq = _build_psin_to_rhotn(dd, time_index)

    sign_Ip = sign(dd.equilibrium.time_slice[time_index].global_quantities.ip)

    # ── output rhotn grid ────────────────────────────────────────────────────
    # If the dd already has a rho grid (e.g. from ZIPFIT/FUSE init), use it.
    # This prevents grid size mismatches when overlaying pfile data onto an
    # existing dd. Otherwise build from ne's psinorm grid (OMFIT default).
    p1d = dd.core_profiles.profiles_1d[time_index]

    rhotn_out = try
        rho = p1d.grid.rho_tor_norm
        length(rho) > 1 ? rho : nothing
    catch
        nothing
    end

    if isnothing(rhotn_out)
        # build from ne psinorm (standalone mode — no existing dd grid)
        rhotn_out = if haskey(qs, "ne")
            psin_ne, _ = _dedup_monotone(qs["ne"].psinorm, qs["ne"].data)
            itp = linear_interpolation(psin_eq, rhotn_eq; extrapolation_bc=Line())
            rho = itp.(clamp.(psin_ne, psin_eq[1], psin_eq[end]))
            sort(rho)   # guarantee monotonic after psinorm→rhotn mapping
        else
            rhotn_eq
        end
        p1d.grid.rho_tor_norm = rhotn_out
    end
    # if dd already has a grid we do NOT overwrite it — just interpolate onto it

    # Helper: interpolate quantity onto rhotn_out, return zeros if key missing
    function get_q(key::String; factor::Float64=1.0) :: Vector{Float64}
        !haskey(qs, key) && return zeros(length(rhotn_out))
        q = qs[key]
        vals = _interp_to_rhotn(q.psinorm, q.data, psin_eq, rhotn_eq, rhotn_out)
        return factor .* vals
    end

    # Helper: use IMAS.ion_element! which sets z_n, a, label, and multiple_states_flag
    # fast=true marks beam/fast ions, fast=false for thermal ions
    function set_element!(ion, z_n, a; fast::Bool=false)
        IMAS.ion_element!(ion, Int(z_n), Float64(a); fast=fast)
    end

    # ── electrons ────────────────────────────────────────────────────────────
    p1d.electrons.density_thermal = get_q("ne"; factor=1e20)   # [m^-3]
    p1d.electrons.temperature     = get_q("te"; factor=1e3)    # [eV]
    # Note: electron rotation fields (omevb, omepp) are OMFIT extensions not
    # present in standard IMASdd — skipped intentionally.

    # ── pressures ────────────────────────────────────────────────────────────
    ptot = get_q("ptot"; factor=1e3)                            # [Pa]
    p1d.pressure_perpendicular = ptot ./ 3.0
    p1d.pressure_parallel      = ptot ./ 3.0

    # ── ExB rotation ─────────────────────────────────────────────────────────
    # sign_Ip because pFile uses abs(Bp) in the definition of omgeb
    p1d.rotation_frequency_tor_sonic = get_q("omgeb"; factor=sign_Ip * 1e3)  # [rad/s]

    # ── radial electric field ─────────────────────────────────────────────────
    if haskey(qs, "er")
        p1d.e_field.radial = get_q("er"; factor=1e3)            # [V/m]
    end

    # ── ion species ──────────────────────────────────────────────────────────
    # OMFIT ordering: impurities first, then [main_ion, beam_ion] at the end.
    nza = pfile.ion_NZA

    if nza !== nothing
        nions_total  = size(nza, 1)
        resize!(p1d.ion, nions_total)
        n_impurities = nions_total - 2    # last two rows are main + beam

        # ── impurity ions (1-based IMAS indexing) ────────────────────────────
        for i in 1:n_impurities
            ion = p1d.ion[i]
            set_element!(ion, Int(nza.Z[i]), nza.A[i])
            ion.multiple_states_flag     = 0
            nz = get_q("nz$(i)"; factor=1e20)
            ion.density_thermal          = nz
            ion.density                  = nz   # no fast component for impurities
            ion.temperature              = get_q("ti"; factor=1e3)

            if Int(nza.Z[i]) == 6   # Carbon — has rotation in pfile
                ion.velocity.toroidal    = get_q("vtor1"; factor=1e3)
                ion.velocity.poloidal    = get_q("vpol1"; factor=1e3)
            else
                ion.velocity.toroidal    = get_q("vtor$(i)"; factor=1e3)
                ion.velocity.poloidal    = get_q("vpol$(i)"; factor=1e3)
            end
        end

        # ── main thermal ion (second-to-last row) ─────────────────────────────
        mk  = n_impurities + 1
        ion = p1d.ion[mk]
        set_element!(ion, Int(nza.Z[end-1]), nza.A[end-1])
        ion.multiple_states_flag     = 0
        ni = get_q("ni"; factor=1e20)
        nb = get_q("nb"; factor=1e20)
        ion.density_thermal          = ni
        ion.density_fast             = nb        # same D species — needed for quasineutrality
        ion.density                  = ni .+ nb  # total D = thermal + fast
        ion.temperature              = get_q("ti";    factor=1e3)
        ion.velocity.toroidal        = get_q("vtor1"; factor=1e3)
        ion.velocity.poloidal        = get_q("vpol1"; factor=1e3)

        # ── beam / fast ion (last row) ────────────────────────────────────────
        bk  = n_impurities + 2
        pb  = get_q("pb"; factor=1e3)                           # [Pa]
        ion = p1d.ion[bk]
        set_element!(ion, Int(nza.Z[end]), nza.A[end]; fast=true)
        ion.multiple_states_flag          = 0
        nb_beam = get_q("nb"; factor=1e20)
        ion.density_fast                  = nb_beam
        ion.density                       = nb_beam  # beam slot: fast only
        ion.pressure_fast_perpendicular   = (2.0/3.0) .* pb
        ion.pressure_fast_parallel        = (1.0/3.0) .* pb

    else
        # ── fallback: no N Z A block → assume D main + C impurity + D beam ───
        @warn "No 'N Z A' block in pfile – assuming D main ion, C impurity, D beam"
        resize!(p1d.ion, 3)

        # Carbon impurity
        set_element!(p1d.ion[1], 6, 12.0)
        p1d.ion[1].multiple_states_flag = 0
        p1d.ion[1].density_thermal      = haskey(qs, "nz1") ?
                                              get_q("nz1"; factor=1e20) :
                                              get_q("ni";  factor=1e-6)   # trace
        p1d.ion[1].temperature          = get_q("ti";    factor=1e3)
        p1d.ion[1].velocity.toroidal    = get_q("vtor1"; factor=1e3)
        p1d.ion[1].velocity.poloidal    = get_q("vpol1"; factor=1e3)

        # Main ion: D
        set_element!(p1d.ion[2], 1, 2.0)
        p1d.ion[2].multiple_states_flag = 0
        p1d.ion[2].density_thermal      = get_q("ni";    factor=1e20)
        p1d.ion[2].temperature          = get_q("ti";    factor=1e3)
        p1d.ion[2].velocity.toroidal    = get_q("vtor1"; factor=1e3)
        p1d.ion[2].velocity.poloidal    = get_q("vpol1"; factor=1e3)

        # Beam: D
        pb = get_q("pb"; factor=1e3)
        set_element!(p1d.ion[3], 1, 2.0; fast=true)
        p1d.ion[3].multiple_states_flag         = 0
        p1d.ion[3].density_fast                 = get_q("nb"; factor=1e20)
        p1d.ion[3].pressure_fast_perpendicular  = (2.0/3.0) .* pb
        p1d.ion[3].pressure_fast_parallel       = (1.0/3.0) .* pb
    end

    return dd
end

end  # module PFileIMAS
