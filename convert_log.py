import codecs

with codecs.open('train_output.txt', 'r', 'utf-16') as f_in:
    content = f_in.read()

with open('train_output_utf8.txt', 'w', encoding='utf-8') as f_out:
    f_out.write(content)

print("Conversion complete!")
