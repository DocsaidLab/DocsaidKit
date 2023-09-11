setup_cfg_path = "setup.cfg"
output_path = "requirements_extras.txt"

# Read the contents of setup.cfg
with open(setup_cfg_path, 'r') as f:
    lines = f.readlines()

# Find the lines after 'options.extras_require'
start_index = None
for i, line in enumerate(lines):
    if 'options.extras_require' in line:
        start_index = i + 1
        break

if start_index is None:
    raise ValueError("options.extras_require not found in setup.cfg")

# Extract the relevant lines until the next section (i.e., line starting with '[')
extracted_lines = []
for line in lines[start_index:]:
    line = line.strip()
    if line.startswith('['):
        break
    if line:  # Check if line is not empty
        extracted_lines.append(line)

# Filter out the line with "torch ="
extracted_lines = [line for line in extracted_lines if line != "torch ="]

# Write the updated extracted lines to requirements_extras.txt
with open(output_path, 'w') as f:
    f.write("\n".join(extracted_lines))
