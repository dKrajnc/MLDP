#include <Evaluation/RandomForestModel.h>
#include <QDebug>

namespace lpmleval
{

//-----------------------------------------------------------------------------

RandomForestModel::RandomForestModel( QSettings* aSettings )
:
	AbstractModel( aSettings ),
	mDecisionTreeModels(),
	mTreeSelection(),
	mNumberSelectedTrees( 0 ),
	mSubsamples()
{
	mTreeSelection = "None";
	bool isValidTreeSelectionMethod;
	bool isValidNumberSelectedTrees;

	if ( aSettings != nullptr )
	{
		mTreeSelection = aSettings->value( "Optimizer/TreeSelection" ).toString(); // Sometimes needed in Optimizer, sometimes in model
		mNumberSelectedTrees = aSettings->value( "Optimizer/NumberSelectedTrees" ).toInt( &isValidNumberSelectedTrees ); // Sometimes needed in Optimizer, sometimes in model

		if ( !isValidNumberSelectedTrees )	qDebug() << "Cannot read setting file for RandomForestModel.";
	}	
}

//-----------------------------------------------------------------------------

RandomForestModel::~RandomForestModel()
{
	for ( auto model : mDecisionTreeModels )
	{
		delete model;
	}

	mDecisionTreeModels.clear();
	mSubsamples.clear();
}

//-----------------------------------------------------------------------------

void RandomForestModel::addDecisionTreeModel( lpmleval::DecisionTreeModel* aTreeModel )
{
	mDecisionTreeModels.append( aTreeModel );
}

//-----------------------------------------------------------------------------

QVariant RandomForestModel::evaluate( const QVector< double >& aFeatureVector )
{
	QVariant predictedLabel;
	QList< QVariant > predictedLabelPerTree;

	//if ( mTreeSelection == "kde" )	QVector< DecisionTreeModel* > mDecisionTreeModels = selectBestTreesByKDE( aFeatureVector );

	// Make prediction with each decision tree
	for ( auto model : mDecisionTreeModels )
	{
		QVariant result = model->evaluate( aFeatureVector );
		predictedLabelPerTree.append( result );
	}

	predictedLabel = calculateMode( predictedLabelPerTree );

	return predictedLabel;
}

//-----------------------------------------------------------------------------

QVariant RandomForestModel::calculateMode( const QVariantList& aList )
{
	int maxFrequency = 0;
	QVariant mostCommonValue;
	QMap< QVariant, int > valueAndFrequency;
	for ( auto value = aList.begin(); value != aList.end(); value++ )
	{
		valueAndFrequency[ *value ]++;
		if ( valueAndFrequency[ *value ] > maxFrequency )
		{
			maxFrequency = valueAndFrequency[ *value ];
			mostCommonValue = *value;
		}
	}

	return mostCommonValue;
}

//-----------------------------------------------------------------------------

}
