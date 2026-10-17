# Source separately from the VASP executable environment.
export VASP_PAW_ROOT=/home/u3129/.local/share/vasp/pseudopotentials/64-b3b00158ad69
export VASP_PAW_REGISTRY="$VASP_PAW_ROOT/registry.json"
vasp_potcar() {
    python3 "$VASP_PAW_ROOT/vasp_potentials.py" assemble "$VASP_PAW_REGISTRY" "$@"
}
