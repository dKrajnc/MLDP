#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractModel.h>
#include <DataRepresentation/TabularData.h>
#include <QSettings>
#include <QVariant>
#include <QString>

namespace lpmleval
{

//-----------------------------------------------------------------------------

struct Node
{
	int splittingFeature;						//!< Column index of the splitting feature
	double splittingValue;						//!< Value of the splitting feature
	QVariant label = "NONE";					//!< "NONE" signals unlabeled nodes. Others are labeled leaf nodes //E
	QMap< QVariant, double > instanceWeights;	//!< Map mapping weights to correspodning instance keys
	Node* left = nullptr;						//!< Left offspring node
	Node* right = nullptr;						//!< Right offspring node

	~Node()
	{
		instanceWeights.clear();
	}

	void save( QDataStream& aOut )
	{
		aOut << qint32( splittingFeature )
			 << splittingValue
			 << label
			 << instanceWeights;
		if ( left != nullptr )
		{
			aOut << QString( "ValidLeft" );
			left->save( aOut );
		}
		else
		{
			aOut << QString( "InvalidLeft" );
		}
		if ( right != nullptr )
		{
			aOut << QString( "ValidRight" );
			right->save( aOut );
		}
		else
		{
			aOut << QString( "InvalidRight" );
		}

	}

	void load( QDataStream& aIn )
	{
		QString validLeft;
		QString validRight;
		qint32 splittingFeature32;

		aIn >> splittingFeature32
			>> splittingValue
			>> label
			>> instanceWeights;

		aIn	>> validLeft;
		if ( validLeft == "ValidLeft" )
		{
			left = new lpmleval::Node;
			left->load( aIn );
		}
		aIn >> validRight;
		if ( validRight == "ValidRight" )
		{
			right = new lpmleval::Node;
			right->load( aIn );
		}

		splittingFeature = int( splittingFeature32 );
	}
};

//-----------------------------------------------------------------------------

class Evaluation_API DecisionTreeModel: public AbstractModel
{

public:

	DecisionTreeModel( QSettings* aSettings );

	~DecisionTreeModel();

	void set( const QVector< double >& aParameters ) override;

	QVariant evaluate( const QVector< double >& aFeatureVector );

	int inputCount() override { return 0; };  // TODO!

	const Node* rootNode() const { return mRootNode; }

	void setRootNode( Node* node );

	void save( QDataStream& aOut ) override
	{	
		aOut << mFeatureNames
			 << static_cast< quint32 >( mNumericType );
		mRootNode->save( aOut );
	}

	void load( QDataStream& aIn )
	{
		quint32 numericType;
		recursiveDeleteTree( mRootNode );

		mRootNode = new lpmleval::Node();

		aIn >> mFeatureNames
			>> numericType;
		mRootNode->load( aIn );

		mNumericType = static_cast< lpmleval::NumericType >( numericType );
	}

private:

	DecisionTreeModel();
	void recursiveDeleteTree( Node* aNode );

private:
	Node* mRootNode;	//!< Root node of the model, holding the information for all subsequent nodes

};

//-----------------------------------------------------------------------------

}
