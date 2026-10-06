using SafeTestsets

files = filter(f -> occursin(".jl", f) && f ≠ "runtests.jl", readdir())
include.(files)
