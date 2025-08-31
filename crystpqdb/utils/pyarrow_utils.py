import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

def get_listArray_struct_fields(array: pa.Array, fields: list[str]) -> list[str]:
    
    if hasattr(array, "combine_chunks"):
        array = array.combine_chunks()
    arr_offsets = array.offsets
    flattened_array = pc.list_flatten(array)
    field_arrays = {}
    for field in fields:
        extracted_array = pc.struct_field(flattened_array,field)
        
        if hasattr(extracted_array, "combine_chunks"):
            extracted_array = extracted_array.combine_chunks()
        new_array = pa.ListArray.from_arrays(arr_offsets, extracted_array)
        field_arrays[field] = new_array
    return field_arrays