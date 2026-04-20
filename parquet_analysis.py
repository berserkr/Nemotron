import pyarrow.parquet as pq

table = pq.read_table("/mnt/vast/proj/checkpoints/bathen/datasets/sft/nemo_super3_8k/splits/train/shard_000000.parquet")
row = table.slice(0, 1).to_pydict()

input_ids = row["input_ids"][0]
loss_mask = row["loss_mask"][0]
seq_start_id = row["seq_start_id"][0]

print("len(input_ids):", len(input_ids))
print("sum(loss_mask):", sum(loss_mask))
print("first_64_loss_mask:", loss_mask[:64])
print("seq_start_id[:20]:", seq_start_id[:20])
print("seq_start_id[-20:]:", seq_start_id[-20:])
