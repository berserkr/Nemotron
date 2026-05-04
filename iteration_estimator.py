import pyarrow.parquet as pq                                                                                                                                                                                                                                                                
from pathlib import Path                                                                                                                                                                                                                                                                    

BS=64
train_dir = Path('/mnt/vast/proj/checkpoints/bathen/datasets/sft/v1_sampled_7m_balanced_ash_128k_8b_cp2/splits/train')                                                                                                                                                                   
total_rows = sum(pq.read_metadata(f).num_rows for f in train_dir.glob('*.parquet'))                                                                                                                                                                                                         
print(f'Total packed sequences: {total_rows}')                                                                                                                                                                                                                                              
print(f'Iters per epoch (GBS={BS}): {total_rows // BS}')                                                                                                                                                                                                                                    
print(f'2 epochs: {2 * (total_rows // BS)}')
print(f'Warmup steps (1 epoch): {int(0.05 * (total_rows // BS))}')     
