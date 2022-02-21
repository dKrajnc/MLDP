/********************************************************************************
** Form generated from reading UI file 'SingleViewer.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SINGLEVIEWER_H
#define UI_SINGLEVIEWER_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SingleViewerUi
{
public:
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QVBoxLayout *verticalLayout;
    QLabel *canvasViewer;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_6;
    QSlider *sliceSlider;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QMenuBar *menuBar;

    void setupUi(QMainWindow *SingleViewerUi)
    {
        if (SingleViewerUi->objectName().isEmpty())
            SingleViewerUi->setObjectName(QString::fromUtf8("SingleViewerUi"));
        SingleViewerUi->resize(1028, 848);
        centralWidget = new QWidget(SingleViewerUi);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        canvasViewer = new QLabel(centralWidget);
        canvasViewer->setObjectName(QString::fromUtf8("canvasViewer"));
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(canvasViewer->sizePolicy().hasHeightForWidth());
        canvasViewer->setSizePolicy(sizePolicy);
        QFont font;
        font.setBold(false);
        font.setWeight(50);
        canvasViewer->setFont(font);
        canvasViewer->setLayoutDirection(Qt::LeftToRight);
        canvasViewer->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(canvasViewer);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_6 = new QLabel(centralWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_2->addWidget(label_6);

        sliceSlider = new QSlider(centralWidget);
        sliceSlider->setObjectName(QString::fromUtf8("sliceSlider"));
        sliceSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(sliceSlider);


        verticalLayout->addLayout(horizontalLayout_2);


        gridLayout->addLayout(verticalLayout, 0, 0, 1, 1);

        SingleViewerUi->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(SingleViewerUi);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        SingleViewerUi->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(SingleViewerUi);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        SingleViewerUi->setStatusBar(statusBar);
        menuBar = new QMenuBar(SingleViewerUi);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1028, 21));
        SingleViewerUi->setMenuBar(menuBar);

        retranslateUi(SingleViewerUi);

        QMetaObject::connectSlotsByName(SingleViewerUi);
    } // setupUi

    void retranslateUi(QMainWindow *SingleViewerUi)
    {
        SingleViewerUi->setWindowTitle(QApplication::translate("SingleViewerUi", "TestApplication", nullptr));
        canvasViewer->setText(QApplication::translate("SingleViewerUi", "TextLabel", nullptr));
        label_6->setText(QApplication::translate("SingleViewerUi", "Slice:", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SingleViewerUi: public Ui_SingleViewerUi {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SINGLEVIEWER_H
