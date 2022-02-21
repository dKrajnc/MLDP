/*!
* \file
* FeatureSelection class defitition. This file is part of Evaluation module.
* The FeatureSelection is a class for describing tabular data manipulation in the feature space by performing the pre-selection of N highest ranking variables
*
* \remarks
*
* \authors
* dKrajnc
*/

#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractTDPAction.h>
#include <QDebug>
#include <qmath.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

/*!
* \brief FeatureSelection class for feature ranking and selection
*/
class Evaluation_API FeatureSelection: public AbstractTBPAction
{
	
public: 

	/*!
	* \brief Constructor to load settings parameters
	* \param [in] aSettings The settigns file
	*/
	FeatureSelection( QSettings* aSettings )
	:
		AbstractTBPAction( aSettings ),
		mFeatureCount( 0 ),
		mRankMethod(),
		mFeatureRanks(),
		mParameters(),
		mSelectedFeatures()
	{
		if ( mSettings == nullptr )
		{
			qDebug() << "FS - Error: Settings is a nullptr";
			mIsInitValid = false;
		}
		else
		{
			bool isFeatureCount;
			mFeatureCount = std::abs( mSettings->value( "FeatureSelection/featureCount" ).toInt( &isFeatureCount ) );
			if ( !isFeatureCount )
			{
				qDebug() << "FS - Error: Invalid parameter featureCount";
				mIsInitValid = false;
			}
			
			mRankMethod = mSettings->value( "FeatureSelection/rankMethod" ).toString();
			if ( mRankMethod == "" )
			{
				qDebug() << "FS - Error: Invalid parameter rankMethod";
				mIsInitValid = false;
			}

			mParameters.insert( "FeatureSelection/featureCount", mFeatureCount );
			mParameters.insert( "FeatureSelection/rankMethod", mRankMethod );
		}
	}

	/*!
	* \brief Destructor
	*/
	~FeatureSelection() {};

public:

	/*!
	* \brief Builds the algorithm based on the input datapackage to calculate feature ranks
	* \param [in] aDataPackage The package of feature and label data
	*/
	void build( const lpmldata::DataPackage& aDataPackage ) override;

	/*!
	* \brief Transforms the datapackage based on the calculated feature ranks
	* \param [in] aDataPackage The package of feature and label data
	* \return lpmldata::DataPackage the transformed datapackage
	*/
	lpmldata::DataPackage run( const lpmldata::DataPackage& aDataPackage ) override;

	/*!
	* \brief Unique class ID
	* \return QString of class ID
	*/
	QString id() override { return "FS"; }

	/*!
	* \brief Algorithm hyperparameters
	* \return QMap < QString, QVariant > of hyperparameter names and values
	*/
	QMap < QString, QVariant > parameters() override { return mParameters; }

	/*!
	* \brief Selected features
	* \return QStringList& names of selected features
	*/
	QStringList& getFeatureNames() { return mSelectedFeatures; };


private:

	QMap< QString, double > rSquaredRank( const lpmldata::DataPackage& aDataPackage );

private:

	int mFeatureCount;
	QString mRankMethod;
	QStringList mSelectedFeatures;
	QMap< QString, double > mFeatureRanks;
	QMap< QString, QVariant > mParameters;
};

}