/*!
* \file
* RandomForest class defitition.
*
* \remarks
*
* \authors
* cspielvogel
*/

#pragma once

#include <Evaluation/AbstractOptimizer.h>
#include <Evaluation/DecisionTreeModel.h>
#include <Evaluation/RandomForestModel.h>
#include <Evaluation/AbstractAnalytics.h>
#include <QSettings>
#include <QPair>
#include <QVector>
#include <QMap>
#include <QString>
#include <QVariant>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API RandomForestOptimizer: public lpmleval::AbstractOptimizer
{

public:
	//! Constructor
	RandomForestOptimizer( QSettings* aSettings, lpmldata::DataPackage* aDataPackage, lpmleval::AbstractModel* aModel, lpmleval::AbstractAnalytics* aAnalytics );

	//! Destructor
	~RandomForestOptimizer();

	//! Trains a random forest model
	void build();

	//! Adds a decision tree model to the random forest model
	void addDecisionTree( const lpmleval::DecisionTreeModel* );

private:
	//! Calculates the boost weight multiplier for a tree model
	void calculateBoostMultiplier( lpmleval::DecisionTreeModel* aTreeModel, const int aBagIndex );

	//! Returns a Pair of feature and label table of the out-of-bag sample
	QPair< lpmldata::TabularData, lpmldata::TabularData > createOOBSample( const lpmldata::TabularData& aInBagSamples );	//<! Creates feature and label tabular data consisting od the out of bag sample

	//! Computes subsamples for each tree by using bagging. Returns a vector with pairs of feature and label tables for each of the created subsamples
	QVector < QPair< lpmldata::TabularData, lpmldata::TabularData > > createRandomSubsample();

	//! Initializes boost weight multiplier with 1 / training_sample_count
	void initializeBoostMultiplier();

	//! Normalization by division of each value by the size of the input vector
	QVector< double > normalize( QVector< double > aValues );

	/*!
	* \brief Determine whether a tree is part of the best trees by evalulation using out-of-bag samples
	* \param aTreeModel DecisionTreeModel pointer of the tree which will be evaluated
	* \param aTreeScores QMap mapping doubles to DecisionTreeModel pointers. Doubles indicate scores for the corresponding tree models
	* \param aInBagSamples TabularData holding the feature values which were used for the building of the aTreeModel
	*/
	QMap< double, lpmleval::DecisionTreeModel* > selectBestTreesByOOB( lpmleval::DecisionTreeModel* aTreeModel, QMap< double, lpmleval::DecisionTreeModel* > aTreeScores, const lpmldata::TabularData& aInBagSamples );

	virtual QVector< double > result() { return QVector< double >(); }

private:

	lpmldata::DataPackage*               mDataPackage;
	lpmleval::AbstractAnalytics*         mAnalytics;

	int mNumberOfTrees;									//!< Int indicating the number of trees created by the Random Forest ( Tree selection happens afterwards )
	QString mQualityMetric;								//!< QString indicating the homogeneity measurment technique
	int mMaxDepth;										//!< Int indicating the maximum depth of a tree
	int mMinSamplesAtLeaf;								//!< Int indicating the minimum number of samples at each leaf
	int mKDEAttributesPerSplit;							//!< Int indicating the number of attributes which KDE selects for later evaluation with the quality metric
	QString mTreeSelection;								//!< QString indicating the method used for tree selection
	int mNumberSelectedTrees;							//!< Int indicating the number of trees which are selected in total
	QString mBaggingMethod;								//!< QString indicating the used bagging method
	double mBagFraction;								//!< Double indicating the fraction of samples which will be sampled by bootstrap aggregating for building the individual trees
	QString mBoosting;									//!< QString indicating the used boosting method
	QVector< double > mTreeWeights;						//!< List of tree errors calculate like in Mishina et al.
	QVector< QMap< QVariant, double > >  mKeysToWeights; //!< QVector holding the key mapped weights for all bags
	QMap< QVariant, double >             mBoostMultiplier;			//!< QMap associating sample keys with the corresponding boosting multiplier for the weights

};

//-----------------------------------------------------------------------------

}
