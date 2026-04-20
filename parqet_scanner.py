import pyarrow.parquet as pq
import glob, sys

PAD_SEQ_TO_MULT = 4  # your CP size

for path in sorted(glob.glob("/mnt/vast/proj/checkpoints/bathen/datasets/sft/qwen3_8b_128k_retok_v2/splits/train/*.parquet", recursive=True)):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["input_ids", "seq_start_id"])
        for i in range(len(table)):
            input_ids = table.column("input_ids")[i].as_py()
            seq_start_id = table.column("seq_start_id")[i].as_py()
            seq_boundaries = seq_start_id + [len(input_ids)]
            seqlens = [seq_boundaries[j+1] - seq_boundaries[j]
                       for j in range(len(seq_boundaries)-1)]

            # simulate cu_seqlens collation
            cu = [0]
            for length in seqlens:
                if length > 1:
                    cu.append(cu[-1] + length - 1)
            if cu[-1] != (len(input_ids) - seqlens.count(1)):  # rough check
                pass
            # check for duplicates
            if len(cu) != len(set(cu)):
                print(f"DUPLICATE: file={path} row_group={rg} row={i}")
                print(f"  seqlens={seqlens} cu={cu}")
                sys.exit(0)
