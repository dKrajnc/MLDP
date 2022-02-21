#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractOptimizer.h>
#include <Evaluation/DecisionTreeModel.h>
#include <Evaluation/TabularDataFilter.h>
#include <DataRepresentation/DataPackage.h>
#include <FileIo/TabularDataFileIo.h>
#include <QSettings>
#include <QString>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API DecisionTreeOptimizer : public AbstractOptimizer
{

public:

	DecisionTreeOptimizer( QSettings* aSettings, lpmldata::DataPackage* aDataPackage );

	//! Destructor
	~DecisionTreeOptimizer();

	QVector< double > result() { return QVector< double >(); }

	//! Builds Tree Model. Uses the Training data provided in the settings file. Used when building stand-alone decision trees
	void build();

	//! Set the instance weights to the given attribute
	void setInstanceWeights( const QMap< QVariant, double >& aInstanceWeights ) { mInstanceWeights = aInstanceWeights; }

	//! Manually create a model by adding a root node. Intended for debugging purposes
	void setModelManually( Node* aRootNode );

	//! Update weights with a multiplier for each weight
	void updateWeights( const QMap< QVariant, double >& aBaggedWeights, const QMap< QVariant, double >& aBoostMultiplier );

private:
	//! Computes class distribution for an attribute. Returns with the best splitting value for the given attribute
	double distribution( QStringList& aKeys, QVector< QVector< double > >& aProportions, QVector< QVector< QMap< QVariant, double > > >& aDistributions, int aAttributeIndex );

	//! Builds the node graph after a tree has been built. The resulting node network serves as information for the decision tree model
	void createModelStructure( Node* aNode );

	//! Returns the attribute used for splitting
	int getAttribute() { return mAttribute; }

	//! Returns the currently best distribution of labels
	QMap< QVariant, double > getClassDistribution() { return mClassDistribution; }

	//! Returns current best splitting value
	double getSplitPoint() { return mSplitPoint; }


	//! Returns information gain
	double gain( QVector< QMap< QVariant, double > >& aDistribution, double aparentPurity );

	//! Returns gini index
	double gini( QVector< QMap< QVariant, double > >& aDistribution, double aparentPurity );

	//! Returns sum of all elements in list
	inline double listSum( const QList< double >& aValues );

	//! Helper function for using logarithms. Returns 0 for values smaller or equal to zero
	double lnHelper( double aValue );

	//! Return normalized vector. All values are divided by the vector sum
	void normalize( QVector< double >& aVector );

	//! Returns entropy before splitting
	double calculateparentPurity( QVector< QMap< QVariant, double > > aDistribution );

	//! Returns real numbers in the range starting from aStart to (excluding) aEnd
	QVector< int > range( int aStart, int aEnd );

	//!Recursively creates nodes
	void recursivePartitioning( QStringList& aNodeKeys, QMap< QVariant, double >& aLabelWeights, QList< int >& aAttributeIndicesWindow, double aTotalWeight, int aDepth );

	//! Returns a random sample from a vector containing integers
	inline int sample( QVector< int >& aIntVector )
	{
		int samplededInt = aIntVector.last();
		aIntVector.pop_back();

		return samplededInt;
	}

	//! Returns a random sample from a list containing strings
	inline QString sample( QStringList& aStrList )
	{
		QString samplededInt = aStrList.last();
		aStrList.pop_back();

		return samplededInt;
	}

	//! Sets a new map with weights for samples
	void setNextInstanceWeights( const QMap< QVariant, double >& aWeights ) { mInstanceWeights = aWeights; }

	//! Resets the number of random features evaluated at each splitting node
	void setNextRandomFeatures( const int aFeatures ) { mRandomFeatures = aFeatures; }

	//! Splits a set of keys according to splitting attribute and value	
	QVector< QStringList > splitData( QStringList& aKeys );

	//! Returns the sum of vector elements
	inline double vectorSum( const QVector< double >& aValues )
	{
		double sum = 0;
		for ( auto value : aValues )
		{
			sum += value;
		}
		return sum;
	}

protected:

	lpmldata::DataPackage* mDataPackage;

	QString mQualityMetric;							//!< QString indicating quality metric for best split evaluation e.g. "gain"
	int mMaxDepth;									//!< Int indicating maximum depth of tree
	int mMinSamplesAtLeaf;							//!< Int indicating minimum number of samples at each label node
	int mKDEAttributesPerSplit;						//!< Int indicating the number of attributes chosen by KDE for evaluation by quality metric
	QString mFeatureSelection;						//!< QString indicating the used method for feature selection
	int mRandomFeatures;							//!< Int indicating the number of randomly chosen features which will be evaluated at each decision node
	QString mBoosting;								//!< QString indicating the method used for boosting
	QMap< QVariant, double > mInstanceWeights;		//!< Map mapping keys to corresponding instance weights

	int mAttribute;									//!< Integer indicating the index of an attribute
	QMap< QVariant, double > mClassDistribution;	//!< Distribution of class labels in the currently best split
	QVector< double > mProportions;					//!< Number of samples going down left and right decision path in the tree
	double mSplitPoint;								//!< Currently best splitting value
	QVector< lpmleval::DecisionTreeOptimizer* > mSuccessors;	//!< Tree Optimizers representing the nodes before creating the model

	Node* mRoot;									//!< Node object representing the root and starting node of the decision tree model

};

//-----------------------------------------------------------------------------

}
