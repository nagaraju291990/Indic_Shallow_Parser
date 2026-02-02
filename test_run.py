from shallow_parser import shallow_parse
from sys import argv
text = "ट्रेन में होने वाले हर अपराध में इस गाँव का कोई न कोई शामिल मिल जाएगा ।"
# ssf = shallow_parse(text, "hin", 'json')
output_file_format = argv[1]
output = shallow_parse(text, "hin", output_file_format)
if output_file_format != 'ssf':
  print (output)
else:
  print("\n".join(output))
