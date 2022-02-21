/*!
* \file
* IsolationForest class defitition. This file is part of Evaluation module.
* The IsolationForest is a class for outlier detection based on Isloation forest algorithm.
* \remarks
*
* \authors
* dkrajnc
*/

#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractTDPAction.h>
#include <Evaluation/CovarianceMatrix.h>
#include <DataRepresentation/Array2D.h>
#include <QDebug>
#include <qmath.h>
#include <random>
#include <algorithm>


namespace dkeval
{

class Evaluation_API IsolationForest: public AbstractTBPAction
{

public:

	/*!
	* \brief Constructor to load settings parameters
	* \param [in] aSettings The settigns file
	*/
	IsolationForest( QSettings* aSettings )
		:
		AbstractTBPAction( aSettings ),
		mTreesnumber( 0 ),
		mSubsamplingSize( 0 ),
		mRoot( nullptr ),
		mCounter( 0 ),
		mOutliers(),
		mParameters()
	{
		//Create parameters
		if ( mSettings == nullptr )
		{
			qDebug() << "IsolationForest - Error: Settings is a nullptr";
			mIsInitValid = false;
		}
		else
		{
			bool isTreesNumber;
			mTreesnumber = std::abs( mSettings->value( "IsolationForest/treeCount" ).toInt( &isTreesNumber ) );
			if ( !isTreesNumber )
			{
				qDebug() << "IsolationForest - Error: Invalid parameter treeCount";
				mIsInitValid = false;
			}

			mParameters.insert( "IsolationForest/treeCount", mTreesnumber );
		}
	}

	/*!
	* \brief Destructor
	*/
	~IsolationForest();
	
	/*!
	* \brief Builds the algorithm based on the input datapackage to indetify outliers
	* \param [in] aDataPackage The package of feature and label data
	*/
	void build( const lpmldata::DataPackage& aDataPackage ) override;


	/*!
	* \brief Transforms the datapackage based on identified outliers
	* \param [in] aDataPackage The package of feature and label data
	* \return lpmldata::DataPackage the transformed datapackage
	*/
	lpmldata::DataPackage run( const lpmldata::DataPackage& aDataPackage ) override;


	/*!
	* \brief Unique class ID
	* \return QString of class ID
	*/
	QString id() override { return "IF"; }

	/*!
	* \brief Algorithm hyperparameters
	* \return QMap < QString, QVariant > of hyperparameter names and values
	*/
	QMap < QString, QVariant > parameters() override { return mParameters; }

	/*!
	* \brief Get identified outliers
	* \return QStringList of outliers
	*/
	QStringList outliers() { return mOutliers; }

private:

	struct Node
	{
		QVector< double > value;
		std::shared_ptr< Node > left   = nullptr;
		std::shared_ptr< Node > right  = nullptr;
	};

	std::shared_ptr< Node > createNode( QVector< double >& aValue );
	void addNode( std::shared_ptr< Node > aPointer );
	void removeNode( std::shared_ptr< Node > aPointer );
	int pathLenght( double aValue, std::shared_ptr< Node > aRoot );
	void resetCounter();
	double averagePathLenght( QVector< int >& aPathLenghts );
	double averagePath( QMap< QString, double >& aPathLenghts );
	double unsuccessfulSearchLenght( int aInstancesCount );
	double anomalyScore( double aUnsuccessfulSearchLenght, double aAveragePathLenght );
	

private:

	int mTreesnumber;
	int mSubsamplingSize;
	std::shared_ptr< Node > mRoot;
	int mCounter;
	QStringList mOutliers;
	QMap< QString, QVariant > mParameters;
};

}
