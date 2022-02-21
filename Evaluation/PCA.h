/*!
* \file
* PCA class defitition. This file is part of Evaluation module.
* The PCA is a class which describes the principal components analysis algorithm tecnique for dimensionality reduction. This version relies on QR decomposition.
*
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
#include <iostream>


namespace dkeval
{

//-----------------------------------------------------------------------------

/*!
* \brief PCA class for dimensionality reduction
*/
class Evaluation_API PCA: public AbstractTBPAction
{
public:

	/*!
	* \brief Constructor to load settings parameters
	* \param [in] aSettings The settigns file
	*/
	PCA( QSettings* aSettings )
		:
		AbstractTBPAction( aSettings ),
		mPreservationPercentage( 0 ),
		mChoosenEigenvectors(),
		mParameters()
	{
		//Create parameters
		if ( mSettings == nullptr )
		{
			qDebug() << "PCA - Error: Settings is a nullptr";
			mIsInitValid = false;
		}
		else
		{
			bool isPreservationPercentage;
			mPreservationPercentage = std::abs( mSettings->value( "PCA/preservationPercentage" ).toInt( &isPreservationPercentage ) );
			if ( !isPreservationPercentage )
			{
				qDebug() << "PCA - Error: Invalid parameter preservationPercentage";
				mIsInitValid = false;
			}

			mParameters.insert( "PCA/preservationPercentage", mPreservationPercentage );
		}
	}

	/*!
	* \brief Destructor
	*/
	~PCA() {};

	/*!
	* \brief Builds the algorithm based on the input datapackage to calculate eigenvalues and eigenvectors
	* \param [in] aDataPackage The package of feature and label data
	*/
	void build( const lpmldata::DataPackage& aDataPackage );

	/*!
	* \brief Transforms the datapackage based on the built eigenvectors
	* \param [in] aDataPackage The package of feature and label data
	* \return lpmldata::DataPackage the transformed datapackage
	*/
	lpmldata::DataPackage run( const lpmldata::DataPackage& aDataPackage ) override;	


	/*!
	* \brief Unique class ID
	* \return QString of class ID
	*/
	QString id() override { return "PCA"; }

	/*!
	* \brief Algorithm hyperparameters
	* \return QMap < QString, QVariant > of hyperparameter names and values
	*/
	QMap < QString, QVariant > parameters() override { return mParameters; }

	/*!
	* \brief Selected eigenvectors
	* \return QVector< QVector< double > > of selected eigenvectors
	*/
	QVector< QVector< double > > getVectors() { return mChoosenEigenvectors; };


private:

	lpmldata::TabularData recenter( const lpmldata::DataPackage& aDataPackage );
	QMap< double, QVector< double > > eigens( lpmldata::TabularData& aDataBase );
	QVector< double > projection( QVector< double >& aInitial, const QVector< double >& aOrthonormal ) ;
	QVector< double > normalize( QVector< double >& aOrthonormal );
	double mean( const QVector< double >& aFeatureVector );
	double standardDeviation( const QVector< double >& aFeatureVector, double aMeanValue );
	QVector< QVector< double > > featureNormalization( QVector< QVector< double > >& aFeatureVector );

private:

	int mPreservationPercentage;
	QVector< QVector< double > > mChoosenEigenvectors;
	QMap< QString, QVariant > mParameters;
};

//-----------------------------------------------------------------------------
}
