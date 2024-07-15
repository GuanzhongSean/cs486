all: clean zip

zip:
	@echo "Zipping..."
	mkdir j76xiao/
	cp -r *.ipynb DQN/ j76xiao/
	rm -rf j76xiao/DQN/utils/__pycache__/ j76xiao/DQN/.ipynb_checkpoints/ j76xiao/DQN/.DS_Store
	jupyter nbconvert --to pdf j76xiao/MDP.ipynb
	jupyter nbconvert --to pdf j76xiao/RL.ipynb
	jupyter nbconvert --to pdf j76xiao/DQN/DQN.ipynb
	zip -r a2.zip j76xiao/

clean:
	@echo "Cleaning..."
	rm -rf a2.zip j76xiao/
