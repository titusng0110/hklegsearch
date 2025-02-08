mkdir -p output
unzip lega.zip -d output
unzip legb.zip -d output
unzip legc.zip -d output
mv output/cap*/*.xml output/
rm -r output/cap*/
