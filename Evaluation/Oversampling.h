/*!
* \file
* Oversampling class defitition. This file is part of Evaluation module.
* The Oversampling is a class for random sample population as well as synthetic sample generation based on methods such as SMOTE, BSMOTE, ADASYN etc.
*
* \remarks
*
* \authors
* dkrajnc
*/

#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractTDPAction.h>
#include <QDebug>
#include <qmath.h>
#include <random>
#include <algorithm>

namespace dkeval
{

class Evaluation_API Oversampling: public AbstractTBPAction
{

public:
	
	/*!
	* \brief Constructor to load settings parameters
	* \param [in] aSettings The settigns file
	*/
	Oversampling( QSettings* aSettings )
		:
		AbstractTBPAction( aSettings ),
		mNeighboursNumber( 0 ), //k1
		mM_NeighboursNumber( 0 ), //k2
		mN_NeighboursNumber( 0 ), //k3
		mOversamplingAmount( 0 ),
		mSamplesDifference( 0 ),
		mSelectionThreshold( 0.0 ),
		mMethod(),
		mLabel( 0 ),
		mSyntheticSamples(),
		mSyntheticPairs(),
		mSyntheticNames(),
		mAutomatic( false ),
		mParameters(),
		mDataPackage( nullptr )
	{
		//Create parameters
		if ( mSettings == nullptr )
		{
			qDebug() << "Oversampling - Error: Settings is a nullptr";
			mIsInitValid = false;
		}
		else
		{
			bool isNeighboursNumber; //used in SMOTE, BSMOTE and MWMOTE (k1)
			mNeighboursNumber = std::abs( mSettings->value( "Oversampling/neighboursNumber" ).toInt( &isNeighboursNumber ) );
			if ( !isNeighboursNumber )
			{
				qDebug() << "Oversampling - Error: Invalid parameter neighboursNumber";
				mIsInitValid = false;
			}

			bool isM_NeighboursNumber; //used in BSMOTE and MWMOTE (k2)
			mM_NeighboursNumber = std::abs( mSettings->value( "Oversampling/m_neighboursNumber" ).toInt( &isM_NeighboursNumber ) );
			if ( !isM_NeighboursNumber )
			{
				qDebug() << "Oversampling - Error: Invalid parameter m_neighboursNumber";
				mIsInitValid = false;
			}

			bool isN_NeighboursNumber; //used in MWMOTE (k3)
			mN_NeighboursNumber = std::abs( mSettings->value( "Oversampling/n_neighboursNumber" ).toInt( &isN_NeighboursNumber ) );
			if ( !isN_NeighboursNumber )
			{
				qDebug() << "Oversampling - Error: Invalid parameter n_neighboursNumber";
				mIsInitValid = false;
			}

			bool isOversamplingCount;
			mOversamplingAmount = std::abs( mSettings->value( "Oversampling/oversamplingPercentage" ).toInt( &isOversamplingCount ) );
			if ( !isOversamplingCount )
			{
				qDebug() << "Oversampling - Error: Invalid parameter oversamplingPercentage";
				mIsInitValid = false;
			}			

			bool isMethod;
			mMethod = mSettings->value( "Oversampling/type" ).toString();
			if ( mMethod == "" )
			{
				qDebug() << "Oversampling - Error: Invalid parameter type";
			}

			bool isAutimatic;
			mAutomatic = mSettings->value( "Oversampling/auto" ).toBool();	

			mParameters.insert( "Oversampling/neighboursNumber", mNeighboursNumber );
			mParameters.insert( "Oversampling/m_neighboursNumber", mM_NeighboursNumber );
			mParameters.insert( "Oversampling/n_neighboursNumber", mN_NeighboursNumber );
			mParameters.insert( "Oversampling/oversamplingPercentage", mOversamplingAmount );
			mParameters.insert( "Oversampling/type", mMethod );
			mParameters.insert( "Oversampling/auto", mAutomatic );
		}

		std::random_device rd;
		mRng = new std::mt19937( rd() );
	}

	/*!
	* \brief Destructor
	*/
	~Oversampling() { delete mRng; }

	/*!
	* \brief Builds the algorithm based on the input datapackage 
	* \param [in] aDataPackage The package of feature and label data
	*/
	void build( const lpmldata::DataPackage& aDataPackage ) override;

	/*!
	* \brief Transforms the datapackage based on generated samples
	* \param [in] aDataPackage The package of feature and label data
	* \return lpmldata::DataPackage the transformed datapackage
	*/
	lpmldata::DataPackage run( const lpmldata::DataPackage& aDataPackage ) override;

	/*!
	* \brief Unique class ID
	* \return QString of class ID
	*/
	QString id() override { return "OS"; }

	/*!
	* \brief Algorithm hyperparameters
	* \return QMap < QString, QVariant > of hyperparameter names and values
	*/
	QMap < QString, QVariant > parameters() override { return mParameters; }

private:

	QMap< double, QString > getNeighbours( const QString& aSampleKey, const QStringList& aSampleKeys );
	QMap< double, QString > getNearestNeighbours( const QString& aSampleKey, const QStringList& aSampleKeys, const int& aNearestNeighboursCount );	
	void filterNearestNeighbours( QMap< double, QString >& aNeighbours, const int& aNeighboursNumber );

	QMap< QString, QVector< double > > generateSyntheticSample( const QString& aSampleKey, const double& aDistance );
	QStringList borderlineMajorities( const QStringList& aMinorityKeys, const QStringList& aMajorityKeys );
	QStringList borderlineMinorities( const QStringList& aMinorityKeys, const QStringList& aMajorityKeys );

	double closenessFactor( const QString& aMajorityKey, const QString& aMinorityKey );
	double densityFactor( const double& aClosenessFactor, const QString& aMajorityKey, const QStringList& aBorderlineMinorities );
	double informationWeight( const QString& aMajorityKey, const QString& aMinorityKey, const QStringList& aBorderlineMinorities );
	double selectionWeight( const QString& aMinorityKey, const QStringList& aBorderlineMinorities, const QStringList& aBorderlineMajorities );
	double averageMinimalDistance( const QStringList& aMinorityKeys );
	double averageDistance( const QStringList& aMinorityKeys );
	double averageClusterDistance( const QStringList& aFirstCluster, const QStringList& aSecondCluster );
	double setThreshold( const double& aAverageDistance );
	
	QVector< QStringList > generateClusters( const QStringList& aMinorityKeys );
	void findClusters( QVector< QStringList >& aInitialClusters );
	QVector< double > toQVectorDouble( const QVariantList& aQVatiantList );


	QMap< double, QString > selectionProbabilities( QMap< double, QString >& aSelectionWeights );
	void smote();
	void bSmote();
	void randomOVersampling();

private:

	int mNeighboursNumber;
	int mM_NeighboursNumber;
	int mN_NeighboursNumber;
	int mOversamplingAmount;
	int mSamplesDifference;
	int mLabel;
	bool mAutomatic;
	double mSelectionThreshold;
	QString mMethod;
	QMap< QString, QVector< double > > mSyntheticSamples;
	QList< QPair< QString, double > > mSyntheticPairs;
	QStringList mSyntheticNames;
	std::mt19937* mRng;
	QMap< QString, QVariant > mParameters;
	const lpmldata::DataPackage* mDataPackage;
};

}
