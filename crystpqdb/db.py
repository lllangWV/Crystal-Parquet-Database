from typing import TypedDict

import pyarrow as pa

site_struct = pa.struct([
    pa.field("species", pa.list_(pa.struct([
        pa.field("element", pa.string()),
        pa.field("occu", pa.int64())
    ]))),
    pa.field("abc", pa.list_(pa.float64())),
    pa.field("xyz", pa.list_(pa.float64())),
    pa.field("properties", pa.struct([
        pa.field("magmom", pa.float64()),
        pa.field("charge", pa.float64()),
        pa.field("forces", pa.list_(pa.float64()))
    ])),
    pa.field("label", pa.string())
]
)

sites_struct = pa.list_(site_struct)

lattice_struct = pa.struct([
    pa.field("matrix", pa.list_(pa.list_(pa.float64()))),
    pa.field("a", pa.float64()),
    pa.field("b", pa.float64()),
    pa.field("c", pa.float64()),
    pa.field("alpha", pa.float64()),
    pa.field("beta", pa.float64()),
    pa.field("gamma", pa.float64()),
    pa.field("pbc", pa.list_(pa.bool_())),
    pa.field("volume", pa.float64())
]
)

symmetry_struct = pa.struct([
    pa.field("crystal_system", pa.string()),
    pa.field("symbol", pa.string()),
    pa.field("number", pa.int32()),
    pa.field("point_group", pa.string()),
    pa.field("symprec", pa.float64()),
    pa.field("angle_tolerance", pa.float64()),
    pa.field("version", pa.string())
])

    
has_props_struct = pa.struct([
    pa.field("materials", pa.bool_()),
    pa.field("thermo", pa.bool_()),
    pa.field("xas", pa.bool_()),
    pa.field("grain_boundaries", pa.bool_()),
    pa.field("chemenv", pa.bool_()),
    pa.field("electronic_structure", pa.bool_()),
    pa.field("absorption", pa.bool_()),
    pa.field("bandstructure", pa.bool_()),
    pa.field("dos", pa.bool_()),
    pa.field("magnetism", pa.bool_()),
    pa.field("elasticity", pa.bool_()),
    pa.field("dielectric", pa.bool_()),
    pa.field("piezoelectric", pa.bool_()),
    pa.field("surface_properties", pa.bool_()),
    pa.field("oxi_states", pa.bool_()),
    pa.field("provenance", pa.bool_()),
    pa.field("charge_density", pa.bool_()),
    pa.field("eos", pa.bool_()),
    pa.field("phonon", pa.bool_()),
    pa.field("insertion_electrodes", pa.bool_()),
    pa.field("substrates", pa.bool_())
])

data_struct = pa.struct([
    pa.field("band_gap", pa.float64()),
    pa.field("band_gap_ind", pa.float64()),
    pa.field("band_gap_dir", pa.float64()),
    pa.field("dos_ef", pa.float64()),
    pa.field("energy_total", pa.float64()),
    pa.field("energy_corrected", pa.float64()),
    pa.field("energy_uncorrected", pa.float64()),
    pa.field("energy_formation", pa.float64()),
    pa.field("energy_above_hull", pa.float64()),
    pa.field("energy_phase_seperation", pa.float64()),
    pa.field("n", pa.float64()),
    pa.field("piezoelectric_modulus", pa.float64()),
    pa.field("e_electronic", pa.float64()),
    pa.field("e_ionic", pa.float64()),
    pa.field("e_total", pa.float64()),
    pa.field("g_reuss", pa.float64()),
    pa.field("g_voigt", pa.float64()),
    pa.field("g_vrh", pa.float64()),
    pa.field("k_reuss", pa.float64()),
    pa.field("k_voigt", pa.float64()),
    pa.field("k_vrh", pa.float64()),
    pa.field("poisson_ratio", pa.float64()),
    pa.field("surface_energy_anisotropy", pa.float64()),
    pa.field("weighted_work_function", pa.float64()),
    pa.field("weighted_surface_energy", pa.float64()),
    pa.field("total_magnetization", pa.float64()),
    pa.field("magnetic_ordering", pa.string()),
    pa.field("stress", pa.list_(pa.list_(pa.float64()))),
    pa.field("is_stable", pa.bool_())
])

structure_struct = pa.struct([
    pa.field("@module", pa.string()),
    pa.field("@class", pa.string()),
    pa.field("lattice", lattice_struct),
    pa.field("sites", sites_struct),
    pa.field("charge", pa.float64())
])


crystpqdb_schema = pa.schema([
    pa.field("source_database", pa.string()),
    pa.field("source_dataset", pa.string()),
    pa.field("source_id", pa.string()),
    pa.field("species", pa.list_(pa.string())),
    pa.field("cart_coords", pa.list_(pa.list_(pa.float64()))),
    pa.field("frac_coords", pa.list_(pa.list_(pa.float64()))),
    pa.field("lattice", lattice_struct),
    pa.field("structure", structure_struct),
    pa.field("data", data_struct),
    pa.field("symmetry", symmetry_struct),
    pa.field("has_props", has_props_struct)
])

