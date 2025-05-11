def summarize_load(loaded_df, name=""):
    print(f"{name} Summary:", 
          f"\tshape: {loaded_df.shape}", 
          f"\tcolumns: {loaded_df.columns.tolist()}",
          sep="\n"
         )