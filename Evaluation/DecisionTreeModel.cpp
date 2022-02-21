#include <Evaluation/DecisionTreeModel.h>
#include <QDebug>

namespace lpmleval
{

//-----------------------------------------------------------------------------

DecisionTreeModel::DecisionTreeModel( QSettings* aSettings )
:
	AbstractModel( nullptr ),
	mRootNode( nullptr )
{
}

//-----------------------------------------------------------------------------

DecisionTreeModel::~DecisionTreeModel()
{
	recursiveDeleteTree( mRootNode );
}

//-----------------------------------------------------------------------------

void DecisionTreeModel::set( const QVector< double >& aParameters )
{
}

//-----------------------------------------------------------------------------

QVariant DecisionTreeModel::evaluate( const QVector< double >& aFeatureVector )
{
	Node* currentNode = mRootNode;

	while ( currentNode->label == "NONE" )
	{
		if ( aFeatureVector[ currentNode->splittingFeature ] < currentNode->splittingValue )
		{
			currentNode = currentNode->left;
		}
		else
		{
			currentNode = currentNode->right;
		}
	}

	QVariant predictedLabel = currentNode->label;

	return predictedLabel;
}

//-----------------------------------------------------------------------------

void DecisionTreeModel::setRootNode( Node* node )
{
	recursiveDeleteTree( mRootNode );

	if ( mRootNode != nullptr )
	{
		delete mRootNode;
		mRootNode = nullptr;
	}

	mRootNode = node;
}

//-----------------------------------------------------------------------------

void DecisionTreeModel::recursiveDeleteTree( Node* aNode )
{
	if ( aNode != nullptr )
	{
		Node* left = aNode->left;
		Node* right = aNode->right;

		delete aNode;
		aNode = nullptr;

		recursiveDeleteTree( left );
		recursiveDeleteTree( right );
	}
}

//-----------------------------------------------------------------------------

}