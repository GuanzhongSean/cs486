all: clean combine zip

combine:
	pdflatex a1.tex
	PYTHONWARNINGS="ignore" jupyter nbconvert --execute --inplace a1q3.ipynb
	jupyter nbconvert --to pdf a1q3.ipynb
	jupyter nbconvert --clear-output --inplace a1q3.ipynb
	pdftk a1.pdf a1q3.pdf cat output a1_combined.pdf

zip:
	zip -r a1.zip a1_combined.pdf a1q3.ipynb *.tex ./*.png a1.py a3.py plots/

clean:
	rm -f a1_combined.pdf a1q3.pdf a1.zip
