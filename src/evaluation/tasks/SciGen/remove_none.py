def remove_none_latex(samples):
  samples = samples.filter(lambda example: example["table_latex"] is not None)
  return 'Provide a textual description of the following table: {{table_latex}}'
