from shallow_parser import shallow_parse

text = "ट्रेन में होने वाले हर अपराध में इस गाँव का कोई न कोई शामिल मिल जाएगा ।"
ssf = shallow_parse(text, "hin", 'json')

print (ssf)
#print("\n".join(ssf))

#print("\n".join(shallow_parse(text, "hin")))

