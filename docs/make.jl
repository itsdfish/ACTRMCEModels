using ACTRMCEModels
using Documenter

DocMeta.setdocmeta!(ACTRMCEModels, :DocTestSetup, :(using ACTRMCEModels); recursive=true)

makedocs(;
    modules=[ACTRMCEModels],
    authors="itsdfish <itsdfish@gmail.com> and contributors",
    sitename="ACTRMCEModels.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
