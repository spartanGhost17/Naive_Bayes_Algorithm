classdef mynb
    methods(Static)
        %% find mean and standard deviation of all features for each class label in training example
        %% get prior probability of each class label (will be used in for posterior prob later)
        %% returned trained model
        function m = fit(train_examples, train_labels)
            %% find all classes in training example
            m.unique_classes = unique(train_labels);
            %% find length of array of unique classes in to use as number of interation in for loop to populate mean and standard dev cells
            m.n_classes = length(m.unique_classes);
            
            %% initialize mean and standard deviation cell arrays fields of our model (m) structure
            m.means = {};
            m.stds = {};
            % start mean and stds population
            %% every class in our training set, find training example with corresponding class label and calculate mean and standard deviation
            %% will be useful in probability distribution computation
            for i = 1:m.n_classes
                %% copy current class label in this_class
				this_class = m.unique_classes(i);
                %% fetch all training examples from current class and copy them in examples_from_this_class
                examples_from_this_class = train_examples{train_labels==this_class,:};
                %% calculate array of means and standards deviations(std) to use later in [probability density calculation]
                m.means{end+1} = mean(examples_from_this_class);
                m.stds{end+1} = std(examples_from_this_class);
            
            end
            % end mean and stds population
            
            %% initialize prior probabilty array
            m.priors = [];
            % start prior probability population
            %% estimate how likely a label is to occure based on how many labels exist in set
            %% for all possible class labels in dataset calculate the prior probability of every training data associated with said class label
            for i = 1:m.n_classes
                %% copy current class label in this_class
				this_class = m.unique_classes(i);
                %% fetch all training examples for current class and copy them in examples_from_this_class
                examples_from_this_class = train_examples{train_labels==this_class,:};
                %% calculate the probability that a randomly chosen example in examples_from_this_class is likely to be part of current class 
                m.priors(end+1) = size(examples_from_this_class,1) / size(train_labels,1);
            
			end
            % end prior probability population
        end
        
        %% Compute a likelihood given each class, for the new example we are trying to classify
        %% Multiply each likelihood by the prior for the corresponding class to give a value proportional to the posterior probability
        %% Generate a prediction of the class label by finding the class with the largest resulting value
        %% return a categorical array of predictions  
        function predictions = predict(m, test_examples)
            %% initialize categorical array
            predictions = categorical;
            %% for every row of test example 
            for i=1:size(test_examples,1)

				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                %% copy test example at current row index
                this_test_example = test_examples{i,:};
                %% calculate the likelihood of current example to classify, using each class label stored in (m.priors)
                this_prediction = mynb.predict_one(m, this_test_example);
                %% add prediction to predictions array
                predictions(end+1) = this_prediction;
            
			end
        end
        
        
        %% Compute a likelihood given each class, for the new example we are trying to classify
        %% calculate prior probability of test example and compute a class label prediction 
        %% return prediction 
        function prediction = predict_one(m, this_test_example)
            %% for every class in our training data set
            for i=1:m.n_classes
                %% find likelihood of current test example, look at each feature value of test example as an independant event (class conditional indepence of naive bayes)
				this_likelihood = mynb.calculate_likelihood(m, this_test_example, i);
                %% get prior probability of current class copy in this_prior
                this_prior = mynb.get_prior(m, i);
                %% calculate and save posterior probability (likelihood * prior probability) of class attached to test example
                posterior_(i) = this_likelihood * this_prior;
            
			end
            %% find the most likely class label(maximum value) in posterior_ array
            [winning_value_, winning_index] = max(posterior_);
            %% use index of winning_value which is the same as index of class in unique_classes array
            prediction = m.unique_classes(winning_index);

        end
        
        %% find likelihood of current test example, since naives bayes can't capture relationships between features,
        %% look at each feature value of test example as an independant event (class conditional indepence of naive bayes)
        function likelihood = calculate_likelihood(m, this_test_example, class)
            
			likelihood = 1;
            %% for every feature value in our training example
			for i=1:length(this_test_example)
                %% since naives bayes can't capture relationships between features, 
                %% we need to capture probability distribution feature by feature and multiply all feature pd together   
                likelihood = likelihood * mynb.calculate_pd(this_test_example(i), m.means{class}(i), m.stds{class}(i));
            end
        end
        %% get prior probability in model (m) using class index (class)
        %% return prior probability fraction
        function prior = get_prior(m, class)
            %% get prior probability index from model using class index 
			prior = m.priors(class);
        
		end
        
        %% calculate probability distribution (a non-zero bell shaped curve), with test example(x), mean of current class features(mu)
        %% stds of current class features (sigma)
        function pd = calculate_pd(x, mu, sigma)
            % mu (center of curve)
            % sigma (width of curve)
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
        
		end
            
    end
end