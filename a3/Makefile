all: clean convert

convert:
	jupyter nbconvert --execute --inplace A3\ Jiaze\ Xiao\ \(j76xiao\).ipynb
	jupyter nbconvert --to pdf A3\ Jiaze\ Xiao\ \(j76xiao\).ipynb
	jupyter nbconvert --clear-output --inplace A3\ Jiaze\ Xiao\ \(j76xiao\).ipynb
	mv A3\ Jiaze\ Xiao\ \(j76xiao\).pdf a3.pdf

submit_%:
	@case $* in \
		q2) marmoset_submit cs486/686 A3Q2 Q2.py;; \
		q3) marmoset_submit cs486/686 A3Q3 pacman.py;; \
		q4) marmoset_submit cs486/686 A3Q4 kmeans.py;; \
		all) echo "Done."; exit 0;; \
		*) echo "Unknown file: $*"; exit 1;; \
	esac

submit_all: submit_q2 submit_q3 submit_q4

clean:
	rm a3.pdf
