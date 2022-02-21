#include <Evaluation/DecisionTreeOptimizer.h>
#include <Evaluation/KernelDensityExtractor.h>
#include <QDebug>

namespace lpmleval
{

//-----------------------------------------------------------------------------

DecisionTreeOptimizer::DecisionTreeOptimizer( QSettings* aSettings, lpmldata::DataPackage* aDataPackage )
:
	AbstractOptimizer( aSettings ),
	mDataPackage( aDataPackage ),
	mQualityMetric(),						
	mMaxDepth( 0 ),						
	mMinSamplesAtLeaf( 0 ),				
	mKDEAttributesPerSplit( 0 ),			
	mFeatureSelection(),					
	mRandomFeatures(),						
	mBoosting(),							
	mInstanceWeights(),		
	mAttribute( 0 ),						
	mClassDistribution(),					
	mProportions(),							
	mSplitPoint( 0.0 ),						
	mSuccessors(),										
	mRoot( nullptr )
{
	// Default values for string based parameters
	mQualityMetric = "gain";
	mFeatureSelection = "random";
	mBoosting = "none";

	bool isValidmMaxDepth;
	bool isValidmMinSamplesAtLeaf;
	bool isValidmKDEAttributesPerSplit;
	bool isValidRandomFeatures;
	bool isValidmSplitPointRandomization;

	// Retrieve parameters from INI file
	mQualityMetric = aSettings->value( "Optimizer/QualityMetric" ).toString().toLower();
	mMaxDepth = aSettings->value( "Optimizer/MaxDepth" ).toInt( &isValidmMaxDepth );
	mMinSamplesAtLeaf = aSettings->value( "Optimizer/MinSamplesAtLeaf" ).toInt( &isValidmMinSamplesAtLeaf );
	mKDEAttributesPerSplit = aSettings->value( "Optimizer/KDEAttributesPerSplit" ).toInt( &isValidmKDEAttributesPerSplit );
	mFeatureSelection = aSettings->value( "Optimizer/FeatureSelection" ).toString().toLower();
	mBoosting = aSettings->value( "Optimizer/Boosting" ).toString().toLower();
	mRandomFeatures = aSettings->value( "Optimizer/RandomFeatures" ).toInt( &isValidRandomFeatures );

	// Check validity of numerical parameters
	if ( !isValidmMaxDepth || !isValidmMinSamplesAtLeaf || !isValidmKDEAttributesPerSplit || !isValidRandomFeatures )
	{
		qDebug() << "Cannot read settings file for DecisionTreeOptimizer.";
	}

	int numFeatures        = mDataPackage->featureCount();
	mKDEAttributesPerSplit = std::min( mKDEAttributesPerSplit, numFeatures );
	mRandomFeatures        = std::min( mRandomFeatures, numFeatures );
}


DecisionTreeOptimizer::~DecisionTreeOptimizer()
{
	mInstanceWeights.clear();
	mClassDistribution.clear();
	mProportions.clear();

	for ( auto successor : mSuccessors )
	{
		delete successor;
	}

	mSuccessors.clear();
}



void DecisionTreeOptimizer::build()
{
	auto keys = mDataPackage->sampleKeys();
	for ( auto key : keys )
	{
		mInstanceWeights.insert( key, 1.0 );
	}

	// Ensure that not more random features are selected than available
	unsigned int numFeatures = mDataPackage->sampleKeys().size();
	if ( mRandomFeatures > numFeatures )
	{
		mRandomFeatures = numFeatures;
	}

	// Set default number of random features
	if ( mRandomFeatures < 1 )
	{
		mRandomFeatures = round( log2( numFeatures ) ) + 1;
		//qDebug() << "Number of random features set to " << mRandomFeatures;
	}

	// Create attribute indices window
	QList< int > attributeIndicesWindow;
	for ( int index = 0; index < mDataPackage->featureCount(); index++ )
	{
		attributeIndicesWindow.append( index );
	}

	// Calculate class weight counts
	double totalWeight = 0;
	double totalSumSquared = 0;
	QMap< QVariant, double > labelWeights;

	for ( auto key : keys )
	{
		QVariant label = mDataPackage->labelDatabase().valueAt( key, mDataPackage->labelIndex() );
		labelWeights[ label ] += mInstanceWeights.value( key );
		totalWeight += mInstanceWeights.value( key );
	}

	// Split data recursively
	recursivePartitioning( keys, labelWeights, attributeIndicesWindow, totalWeight, 0 );

	// Use splitting criteria to create model structure
	mRoot = new Node();
	createModelStructure( mRoot );

	DecisionTreeModel* treeModel = new DecisionTreeModel( nullptr );  // OR mSettings?
	treeModel->setRootNode( mRoot );

	//mModel = new lpmleval::DecisionTreeModel( nullptr );  // OR mSettings?
	if ( mModel != nullptr )
	{
		delete mModel;
		mModel = nullptr;
	}

	mModel = treeModel;

}


void DecisionTreeOptimizer::setModelManually( Node* aRootNode )
{
	if ( mRoot != nullptr )
	{
		delete mRoot;
		mRoot = nullptr;
	}

	mRoot = new Node();

	DecisionTreeModel* treeModel = new DecisionTreeModel( nullptr );  // OR mSettings?
	treeModel->setRootNode( aRootNode );

	if ( mModel != nullptr )
	{
		delete mModel;
		mModel = nullptr;
	}

	//mModel = new lpmleval::DecisionTreeModel( nullptr );  // OR mSettings?
	mModel = treeModel;
}


void DecisionTreeOptimizer::updateWeights( const QMap< QVariant, double >& aBaggedWeights, const QMap< QVariant, double >& aBoostMultiplier )
{
	for ( auto key : aBaggedWeights.keys() )
	{
		if ( mInstanceWeights.contains( key ) )
		{
			mInstanceWeights[ key ] = aBaggedWeights.value( key ) * aBoostMultiplier.value( key );
		}
		else
		{
			qDebug() << "Key" << key << "is not in mInstanceWeights!!!";
		}
		
	}
}

	//! Computes class distribution for an attribute. Returns with the best splitting value for the given attribute
double DecisionTreeOptimizer::distribution( QStringList& aKeys, QVector< QVector< double > >& aProportions, QVector< QVector< QMap< QVariant, double > > >& aDistributions, int aAttributeIndex )
{
	QVariant splittingValue = NULL;

	//Sort keys by ascending attribute value
	QMap< QString, QVariant > keyValueMap;
	QList< QString > orderedKeys;
	for ( auto key : aKeys )
	{
		keyValueMap[ key ] = mDataPackage->featureDatabase().valueAt( key, aAttributeIndex );
	}

	QVector< QPair< QVariant, QString > > mapVector;

	// Insert entries
	for ( auto key : keyValueMap.keys() )
	{
		QPair< double, QString > pair = { keyValueMap[ key ].toDouble(), key };
		mapVector.append( pair );
	}

	std::sort( mapVector.begin(), mapVector.end() );

	for ( auto pair : mapVector )
	{
		orderedKeys.append( pair.second );
	}

	// Calculate weight distribution of labels
	lpmleval::TabularDataFilter filter;
	auto labelOutcomes = filter.labelGroups( mDataPackage->labelDatabase(), mDataPackage->labelIndex() );
	QVector< QMap< QVariant, double > > currentDistribution = { {}, {} };
	for ( auto label : labelOutcomes )  // TODO: Change to dataset, read out the labeloutcomes.
	{
		currentDistribution[ 0 ][ label ] = 0;
	}

	for ( auto key : orderedKeys )
	{
		currentDistribution[ 1 ][ mDataPackage->labelDatabase().valueAt( key, mDataPackage->labelIndex() ) ] += mInstanceWeights.value( key );
	}

	// Calculate purity of sample set before splitting
	double parentPurity = calculateparentPurity( currentDistribution );

	// Copy weight distribution of labels
	QVector< QMap< QVariant, double > > distributionCopy = currentDistribution;

	// Calculate purity for all attribute values
	double currentSplitPoint = mDataPackage->featureDatabase().valueAt( orderedKeys[ 0 ], aAttributeIndex ).toDouble();
	double currentPurity = -DBL_MAX;
	double bestPurity = -DBL_MAX;

	for ( auto key : orderedKeys )
	{
		double attributeValue = mDataPackage->featureDatabase().valueAt( key, aAttributeIndex ).toDouble();

		if ( attributeValue > currentSplitPoint )
		{
			if ( mQualityMetric == "gain" )
			{
				currentPurity = gain( currentDistribution, parentPurity );
			}
			else if ( mQualityMetric == "gini" )
			{
				currentPurity = gini( currentDistribution, parentPurity );
			}
			else
			{
				qDebug() << "Error: Unknown quality metric: " << mQualityMetric;
			}

			// Update best value
			if ( currentPurity > bestPurity )
			{
				bestPurity = currentPurity;
				splittingValue = ( attributeValue + currentSplitPoint ) / 2.0;	// Splitting value inbetween adjactent values of samples

				if ( splittingValue <= currentSplitPoint )
				{
					splittingValue = attributeValue;
				}
				distributionCopy = currentDistribution;
			}
			currentSplitPoint = attributeValue;
		}
		QVariant label = mDataPackage->labelDatabase().valueAt( key, mDataPackage->labelIndex() );
		currentDistribution[ 0 ][ label ] += mInstanceWeights.value( key );
		currentDistribution[ 1 ][ label ] -= mInstanceWeights.value( key );
	}

	// Calculate weights for subsets
	if ( aProportions.isEmpty() )
	{
		aProportions = { {}, {} };
		for ( auto label : labelOutcomes )
		{
			aProportions[ 0 ].append( {} );
			aProportions[ 1 ].append( {} );
		}
	}

	for ( int leftRightIndex = 0; leftRightIndex < 2; leftRightIndex++ )
	{
		aProportions[ 0 ][ leftRightIndex ] = listSum( distributionCopy[ leftRightIndex ].values() );
	}

	if ( vectorSum( aProportions[ 0 ] ) == 0.0 )
	{
		for ( int index = 0; index < aProportions[ 0 ].size(); index++ )
		{
			aProportions[ 0 ][ index ] = 1.0 / aProportions[ 0 ].size();
		}
	}
	else
	{
		normalize( aProportions[ 0 ] );
	}

	if ( aDistributions.isEmpty() )
	{
		aDistributions = { {} };
	}

	aDistributions[ 0 ] = distributionCopy;

	return splittingValue.toDouble();
}

	//! Builds the node graph after a tree has been built. The resulting node network serves as information for the decision tree model
void DecisionTreeOptimizer::createModelStructure( Node* aNode )
{
	if ( getAttribute() == -1 )
	{
		QVariant bestLabel = NULL;
		double highestValue = -DBL_MAX;
		for ( auto element : getClassDistribution().keys() )
		{
			if ( getClassDistribution()[ element ] > highestValue )
			{
				bestLabel = element;
				highestValue = getClassDistribution()[ element ];
			}
		}
		aNode->label = bestLabel;
	}
	else
	{
		aNode->splittingFeature = getAttribute();
		aNode->splittingValue = getSplitPoint();

		for ( auto leftRightIndex : { 0, 1 } )
		{
			Node* node = new Node();
			if ( leftRightIndex == 0 )
			{
				aNode->left = new Node();
				node = aNode->left;
			}
			else
			{
				aNode->right = new Node();
				node = aNode->right;
			}
			mSuccessors[ leftRightIndex ]->createModelStructure( node );
		}
	}
}


	//! Returns information gain
double DecisionTreeOptimizer::gain( QVector< QMap< QVariant, double > >& aDistribution, double aparentPurity )
{
	// Calculate class probabilities for quality metric
	QVector< double > labelSumsLeft;
	QVector< double > labelSumsRight;
	for ( int index = 0; index < aDistribution[ 0 ].size() && index < aDistribution[ 1 ].size(); index++ )
	{
		labelSumsLeft.append( 0 );
		labelSumsRight.append( 0 );
	}

	for ( int labelIndex = 0; labelIndex < aDistribution[ 0 ].size(); labelIndex++ )
	{
		QVariant labelOutcome = mDataPackage->labelOutcomes().at( labelIndex );
		labelSumsLeft[ labelIndex ] += aDistribution[ 0 ][ labelOutcome ];
	}
	for ( int labelIndex = 0; labelIndex < aDistribution[ 1 ].size(); labelIndex++ )
	{
		QVariant labelOutcome = mDataPackage->labelOutcomes().at( labelIndex );
		labelSumsRight[ labelIndex ] += aDistribution[ 1 ][ labelOutcome ];
	}

	QVector< double > classProbabilitiesLeft;
	for ( auto labelFrequency : labelSumsLeft )
	{
		classProbabilitiesLeft.append( labelFrequency / vectorSum( labelSumsLeft ) );
	}
	QVector< double > classProbabilitiesRight;
	for ( auto labelFrequency : labelSumsRight )
	{
		classProbabilitiesRight.append( labelFrequency / vectorSum( labelSumsRight ) );
	}

	double entropyLeft = 0;
	for ( auto probability : classProbabilitiesLeft )
	{
		entropyLeft += probability * lnHelper( probability );
	}
	entropyLeft = -entropyLeft;

	double entropyRight = 0;
	for ( auto probability : classProbabilitiesRight )
	{
		entropyRight += probability * lnHelper( probability );
	}
	entropyRight = -entropyRight;

	double numberSamplesLeft = vectorSum( labelSumsLeft );
	double numberSamplesRight = vectorSum( labelSumsRight );
	double relativeSamplesLeft = numberSamplesLeft / ( numberSamplesLeft + numberSamplesRight );
	double relativeSamplesRight = numberSamplesRight / ( numberSamplesLeft + numberSamplesRight );
	double gain = aparentPurity - ( relativeSamplesLeft * entropyLeft + relativeSamplesRight * entropyRight );

	return gain;
}

	//! Returns gini index
double DecisionTreeOptimizer::gini( QVector< QMap< QVariant, double > >& aDistribution, double aparentPurity )
{
	// Calculate class probabilities for quality metric
	QVector< double > labelSumsLeft;
	QVector< double > labelSumsRight;
	for ( int index = 0; index < aDistribution[ 0 ].size() && index < aDistribution[ 1 ].size(); index++ )
	{
		labelSumsLeft.append( 0 );
		labelSumsRight.append( 0 );
	}

	for ( int labelIndex = 0; labelIndex < aDistribution[ 0 ].size(); labelIndex++ )
	{
		QVariant labelOutcome = mDataPackage->labelOutcomes().at( labelIndex );
		labelSumsLeft[ labelIndex ] += aDistribution[ 0 ][ labelOutcome ];
	}
	for ( int labelIndex = 0; labelIndex < aDistribution[ 1 ].size(); labelIndex++ )
	{
		QVariant labelOutcome = mDataPackage->labelOutcomes().at( labelIndex );
		labelSumsRight[ labelIndex ] += aDistribution[ 1 ][ labelOutcome ];
	}

	QVector< double > classProbabilitiesLeft;
	for ( auto labelFrequency : labelSumsLeft )
	{
		classProbabilitiesLeft.append( labelFrequency / vectorSum( labelSumsLeft ) );
	}
	QVector< double > classProbabilitiesRight;
	for ( auto labelFrequency : labelSumsRight )
	{
		classProbabilitiesRight.append( labelFrequency / vectorSum( labelSumsRight ) );
	}

	double giniLeft = 0;
	for ( auto probability : classProbabilitiesLeft )
	{
		giniLeft += probability * ( 1 - probability );
	}

	double giniRight = 0;
	for ( auto probability : classProbabilitiesRight )
	{
		giniRight += probability * ( 1 - probability );
	}

	double numberSamplesLeft = vectorSum( labelSumsLeft );
	double numberSamplesRight = vectorSum( labelSumsRight );
	double relativeSamplesLeft = numberSamplesLeft / ( numberSamplesLeft + numberSamplesRight );
	double relativeSamplesRight = numberSamplesRight / ( numberSamplesLeft + numberSamplesRight );
	double gini = aparentPurity - ( relativeSamplesLeft * giniLeft + relativeSamplesRight * giniRight );

	return gini;
}

//! Returns sum of all elements in list
inline double DecisionTreeOptimizer::listSum( const QList< double >& aValues )
{
	double sum = 0;
	for ( auto value : aValues )
	{
		sum += value;
	}
	return sum;
}

//! Helper function for using logarithms. Returns 0 for values smaller or equal to zero
double DecisionTreeOptimizer::lnHelper( double aValue )
{
	if ( aValue <= 0 )
	{
		return 0;
	}
	else
	{
		return aValue * log( aValue );
	}
}

	//! Return normalized vector. All values are divided by the vector sum
void DecisionTreeOptimizer::normalize( QVector< double >& aVector )
{
	double vectorSum = 0;
	for ( auto value : aVector )
	{
		vectorSum += value;
	}

	for ( int index = 0; index < aVector.size(); index++ )
	{
		aVector[ index ] /= vectorSum;
	}
}

	//! Returns entropy before splitting
double DecisionTreeOptimizer::calculateparentPurity( QVector< QMap< QVariant, double > > aDistribution )
{
	// Calculate class probability for quality metric
	QVector< double > labelSums;
	for ( int index = 0; index < aDistribution[ 0 ].size() && index < aDistribution[ 1 ].size(); index++ )
	{
		labelSums.append( 0 );
	}

	for ( auto leftRightIndex : { 0, 1 } )
	{
		for ( int labelIndex = 0; labelIndex < aDistribution[ leftRightIndex ].size(); labelIndex++ )
		{
			QVariant labelOutcome = mDataPackage->labelOutcomes().at( labelIndex );
			labelSums[ labelIndex ] += aDistribution[ leftRightIndex ][ labelOutcome ];
		}
	}

	QVector< double > classProbabilities;
	for ( auto labelFrequency : labelSums )
	{
		classProbabilities.append( labelFrequency / vectorSum( labelSums ) );
	}

	// Parent entropy
	if ( mQualityMetric == "gain" )
	{
		double entropy = 0;
		for ( auto probability : classProbabilities )
		{
			entropy += probability * lnHelper( probability );
		}
		entropy = -entropy;
		return entropy;
	}
	// Parent gini index
	else if ( mQualityMetric == "gini" )
	{
		double parentGini = 0;
		for ( auto probability : classProbabilities )
		{
			parentGini += probability * ( 1 - probability );
		}
		return parentGini;
	}
	// Invalid quality metric
	else
	{
		qDebug() << "Error: Unkown quality metric " << mQualityMetric;
		return 0;
	}
}

//! Returns real numbers in the range starting from aStart to (excluding) aEnd
QVector< int > DecisionTreeOptimizer::range( int aStart, int aEnd )
{
	std::random_device rd;
	std::mt19937 generator( rd() );
	std::uniform_int_distribution<> distribution( aStart, aEnd );

	QVector< int > ints;
	int real = aStart;
	while ( real < aEnd )
	{
		ints.append( real );
		real++;
	}

	std::shuffle( ints.begin(), ints.end(), generator );

	return ints;
}

	//!Recursively creates nodes
void DecisionTreeOptimizer::recursivePartitioning( QStringList& aNodeKeys, QMap< QVariant, double >& aLabelWeights, QList< int >& aAttributeIndicesWindow, double aTotalWeight, int aDepth )
{
	// Create a leaf when Node is empty
	if ( aNodeKeys.empty() )
	{
		mAttribute = -1;
		mClassDistribution = { {}, {} };
		mProportions = { {}, {} };

		qDebug() << "Reached empty node!";
		return;
	}

	// Calculate total weight at node
	aTotalWeight = listSum( aLabelWeights.values() );
	QList< double > labelWeights = aLabelWeights.values();
	double highestLabelWeight = *std::max_element( labelWeights.begin(), labelWeights.end() );

	if ( aTotalWeight < 2 * mMinSamplesAtLeaf ||	 // Node size reached
			highestLabelWeight == aTotalWeight ||	 // Only one label
			aDepth == mMaxDepth						 // Max depth reached
			)
	{
		// Create leaf node
		mAttribute = -1;
		mClassDistribution = aLabelWeights;
		mProportions = {};
		return;
	}

	// Calculate class distributions and splitting value for each attribute
	double purity = -DBL_MAX;
	double bestSplit = -DBL_MAX;
	QVector< QMap< QVariant, double > > bestDistributions = { {} };
	QVector< double > bestProportions = {};
	int bestIndex = 0;

	QVector< QVector< double > > proportions;
	QVector< QVector< QMap< QVariant, double > > > distributions;

	int attributeIndex = 0;
	int windowSize = aAttributeIndicesWindow.size();
	int numRandomFeatures = mRandomFeatures;
	bool gainFound = false;
	QVector< int > shuffledInts = range( 0, windowSize );
	bool KDEexecuted = false;

	// Find best attributes for splitting
	lpmleval::TabularDataFilter filter;
	while ( ( windowSize > 0 ) && ( numRandomFeatures-- > 0 || !gainFound ) && !KDEexecuted )
	{
		if ( mFeatureSelection == "kde" )
		{
			// Calculate best attribute by KDE
			QList< int > randomIndices;
			for ( int index = 0; index < numRandomFeatures; index++ )
			{
				int randomIndex = sample( shuffledInts );
				randomIndices.append( randomIndex );
			}

			// Get data set at node and reduce to puritys for randomly chosen attributes
			lpmldata::TabularData featureSet = filter.subTableByKeys( mDataPackage->featureDatabase(), aNodeKeys );
			lpmldata::TabularData labelSet = filter.subTableByKeys( mDataPackage->labelDatabase(), aNodeKeys );
			lpmldata::TabularData attributeReducedFeatureSet = filter.subTableByAttributes( featureSet, randomIndices );

			lpmleval::KernelDensityExtractor kde( attributeReducedFeatureSet, labelSet, mDataPackage->labelIndex() );
			attributeIndex = kde.overlapRatios().values()[ 0 ];	// Get attribute with lowest distribution overlap
			KDEexecuted = true;
		}
		else
		{
			int randomIndex = sample( shuffledInts );
			attributeIndex = aAttributeIndicesWindow[ randomIndex ];

			// Remove chosen attribute index from window
			aAttributeIndicesWindow[ randomIndex ] = aAttributeIndicesWindow[ windowSize - 1 ];
			aAttributeIndicesWindow[ windowSize - 1 ] = attributeIndex;
			windowSize--;
		}

		//Find best split for current attribute
		double currentSplit = distribution( aNodeKeys, proportions, distributions, attributeIndex );

		// Calculate gain for current split
		double currentPurity;
		if ( mQualityMetric == "gini" )
		{
			currentPurity = gini( distributions[ 0 ], calculateparentPurity( distributions[ 0 ] ) );

		}
		else
		{
			currentPurity = gain( distributions[ 0 ], calculateparentPurity( distributions[ 0 ] ) );
		}

		if ( currentPurity > 0 )
		{
			gainFound = true;
		}

		if ( currentPurity > purity || ( currentPurity == purity && attributeIndex < bestIndex ) )
		{
			int mAttributeSave = mAttribute;

			purity = currentPurity;
			bestIndex = attributeIndex;
			bestSplit = currentSplit;
			bestProportions = proportions[ 0 ];
			bestDistributions = distributions[ 0 ];
		}
	}

	mAttribute = bestIndex;
	if ( purity > 0 )
	{
		// Build nodes
		mSplitPoint = bestSplit;
		mProportions = bestProportions;
		QVector< QStringList > subsets = splitData( aNodeKeys );

		for ( auto leftRightIndex : { 0, 1 } )
		{
			lpmleval::DecisionTreeOptimizer* newNode = new lpmleval::DecisionTreeOptimizer( mSettings, mDataPackage );
			newNode->setNextInstanceWeights( mInstanceWeights );
			newNode->setNextRandomFeatures( mRandomFeatures );

			mSuccessors.append( newNode );
			mSuccessors[ leftRightIndex ]->recursivePartitioning( subsets[ leftRightIndex ], bestDistributions[ leftRightIndex ],
																	aAttributeIndicesWindow, 0.0, aDepth + 1 );
		}

		bool successorEmpty = false;
		for ( int index = 0; index < subsets.size(); index++ )
		{
			if ( mSuccessors[ index ]->mClassDistribution.values().isEmpty() )
			{
				successorEmpty = true;
				break;
			}
		}

		if ( successorEmpty )
		{
			mClassDistribution = aLabelWeights;
		}
	}
	else
	{
		// Make a leaf
		mAttribute = -1;
		mClassDistribution = aLabelWeights;
	}
}


QVector< QStringList > DecisionTreeOptimizer::splitData( QStringList& aKeys )
{
	QVector< QStringList > splittedData = { {}, {} };

	// Iterate over samples
	for ( auto key : aKeys )
	{
		splittedData[ mDataPackage->featureDatabase().valueAt( key, mAttribute ).toDouble() < mSplitPoint ? 0 : 1 ].append( key );
		continue;
	}

	return splittedData;
}

};

//-----------------------------------------------------------------------------

