#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractModel.h>
#include <Evaluation/DecisionTreeModel.h>
#include <DataRepresentation/DataPackage.h>
#include <DataRepresentation/TabularData.h>
#include <QSettings>
#include <QVariant>
#include <QString>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API RandomForestModel: public AbstractModel
{

public:

	RandomForestModel( QSettings* aSettings );

	~RandomForestModel();

	void setSubsamples( const QVector < QPair< lpmldata::TabularData, lpmldata::TabularData > >& aSubsamples ) { mSubsamples = aSubsamples; }

	void RandomForestModel::addDecisionTreeModel( lpmleval::DecisionTreeModel* aTreeModel );


	QVariant RandomForestModel::evaluate( const QVector< double >& aFeatureVector );

	void set( const QVector< double >& aParameters ) override {}

	int inputCount() override { return 0; };  // TODO!

	void save( QDataStream& aOut )
	{
		aOut << mFeatureNames
			 << static_cast< quint32 >( mNumericType )
			 << qint32( mDecisionTreeModels.size() );
			for ( auto model : mDecisionTreeModels )
			{
				model->save( aOut );
			}

			aOut << mTreeSelection
				 << qint32( mNumberSelectedTrees );
	}

	void load( QDataStream& aIn )
	{
		quint32 numericType;
		QVector < lpmleval::DecisionTreeModel* > decisionTreeModels;
		int treeSize;

		aIn >> mFeatureNames
			>> numericType
			>> treeSize;

		for ( int i = 0; i < treeSize; ++i )
		{
			lpmleval::DecisionTreeModel* DTModel = new lpmleval::DecisionTreeModel( mSettings );
			DTModel->load( aIn );
			decisionTreeModels.push_back( DTModel );
		}

		aIn >> mTreeSelection
			>> mNumberSelectedTrees;

		mDecisionTreeModels = decisionTreeModels;
		mNumericType = static_cast< lpmleval::NumericType >( numericType );
	}

	/*friend QDataStream& RandomForestModel::operator<<( QDataStream& out, lpmleval::RandomForestModel* aModel )
	{
		lpmleval::AbstractModel* abstractModel = dynamic_cast< lpmleval::AbstractModel* >( aModel );
		if ( abstractModel )
		{
			out << *abstractModel;
			out << qint32( aModel->mDecisionTreeModels.size() );
			for ( auto model : aModel->mDecisionTreeModels )
			{
				out << model;
			}

			out << aModel->mTreeSelection
				<< qint32( aModel->mNumberSelectedTrees );
		}

		return out;
	}

	friend QDataStream& RandomForestModel::operator>>( QDataStream& in, lpmleval::RandomForestModel* aModel )
	{
		lpmleval::AbstractModel* abstractModel = dynamic_cast< lpmleval::AbstractModel* >( aModel );
		if ( abstractModel )
		{
			QVector < lpmleval::DecisionTreeModel* > decisionTreeModels;
			int treeSize;

			in >> *abstractModel
			   >> treeSize;

			for ( int i = 0; i < treeSize; ++i )
			{
				lpmleval::DecisionTreeModel* DTModel = new lpmleval::DecisionTreeModel( aModel->mSettings );
				in >> DTModel;
				decisionTreeModels.push_back( DTModel );				
			}

			aModel->mDecisionTreeModels = decisionTreeModels;

			in >> aModel->mTreeSelection
			   >> aModel->mNumberSelectedTrees;
		}

		return in;
	}*/

private:

	RandomForestModel();

	QVariant calculateMode( const QVariantList& aList );

private:

	QVector < lpmleval::DecisionTreeModel* >                           mDecisionTreeModels;	//!< Vector of Decision tree model pointers which are part of the random forest model
	QString                                                            mTreeSelection;  //!< String indicating the tree selection method (None, OOB or KDE)
	int                                                                mNumberSelectedTrees;  //!< Int indicating the number of trees to be selected by the tree selection method
	QVector < QPair< lpmldata::TabularData, lpmldata::TabularData > >  mSubsamples;	 //!< Vector of feature-label set pairs which were bagged by the optimizer

};

//-----------------------------------------------------------------------------

}
