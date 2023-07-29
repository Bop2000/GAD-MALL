# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2018 replay file
# Internal Version: 2017_11_08-01.21.41 127140
# Run by Administrator on Wed Jun 23 09:21:51 2021
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=273.328125, 
    height=210.518508911133)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
#--- Recover file: 'abaqus1.rec' ---
# -*- coding: mbcs -*- 
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *


# Abaqus Initialization
executeOnCaeStartup()
#session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
#    referenceRepresentation=ON)
Mdb()
#: A new model database has been created.
#: The model "Model-1" has been created.
#session.viewports['Viewport: 1'].setValues(displayedObject=None)
#a = mdb.models['Model-1'].rootAssembly
#session.viewports['Viewport: 1'].setValues(displayedObject=a)
wsDir = 'E:/qinyu/daijiabao/inp/'  # Your directory containing source .inp files here.


def modelSetUp(inputDir, fileName):
    inputFileName = inputDir + fileName + '.inp'
    taskName = fileName

    mdb.ModelFromInputFile(name='abaqusModel8', inputFileName=inputFileName)
    #: The model "abaqusModel8" has been created.
    #: The part "PART-1" has been imported from the input file.
    #: The model "abaqusModel8" has been imported from an input file. 
    #: Please scroll up to check for error and warning messages.
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models['abaqusModel8'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=OFF)
    p1 = mdb.models['abaqusModel8'].parts['PART-1']
    session.viewports['Viewport: 1'].setValues(displayedObject=p1)
    
    # add meterial
    mdb.models['abaqusModel8'].Material(name='Material-1')
    mdb.models['abaqusModel8'].materials['Material-1'].Density(table=((7140, ), ))
    mdb.models['abaqusModel8'].materials['Material-1'].Elastic(table=((30000000000.0, 0.25), ))
    mdb.models['abaqusModel8'].materials['Material-1'].Plastic(table=((
    118000000.0, 0.0), (125000000.0, 0.003), (130000000.0, 0.006), (
    134000000.0, 0.009), (137000000.0, 0.012), (139000000.0, 0.015), (
    141000000.0, 0.018), (143000000.0, 0.024), (145000000.0, 0.036), (
    146000000.0, 0.042), (148000000.0, 0.048), (149000000.0, 0.054), (
    150000000.0, 0.084), (151.0, 0.118)))
    mdb.models['abaqusModel8'].materials['Material-1'].DuctileDamageInitiation(table=((0.12, 0.33, 0.0), ))
    mdb.models['abaqusModel8'].materials['Material-1'].ductileDamageInitiation.DamageEvolution(
    type=DISPLACEMENT, table=((0.12, ), ))

    p1 = mdb.models['abaqusModel8'].parts['PART-1']
    session.viewports['Viewport: 1'].setValues(displayedObject=p1)
    mdb.models['abaqusModel8'].HomogeneousSolidSection(name='Gyroid', material='Material-1', thickness=None)
    p = mdb.models['abaqusModel8'].parts['PART-1']
    e = p.elements
    elements = e
    #elements = e.getSequenceFromMask(mask=('[#ffffffff:1000 #3 ]', ), )
    region = p.Set(elements=elements, name='Gyroid')
    p = mdb.models['abaqusModel8'].parts['PART-1']
    p.SectionAssignment(region=region, sectionName='Gyroid', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
        engineeringFeatures=OFF)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=ON)
    s = mdb.models['abaqusModel8'].ConstrainedSketch(name='__profile__', 
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(5.0, 0.0))
    p = mdb.models['abaqusModel8'].Part(name='Round', dimensionality=THREE_D, 
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['abaqusModel8'].parts['Round']
    p.BaseSolidExtrude(sketch=s, depth=2.0)
    s.unsetPrimaryObject()
    p = mdb.models['abaqusModel8'].parts['Round']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['abaqusModel8'].sketches['__profile__']
    p = mdb.models['abaqusModel8'].parts['Round']
    c1 = p.cells
    p.RemoveCells(cellList = c1[0:1])
    #: 
    #: One shell per selected cell has been created from the cell's faces.
    p = mdb.models['abaqusModel8'].parts['Round']
    p.queryAttributes()
    #: 
    #: Part: "Round",  Modeling space: 3D,  Type: Discrete rigid
    p = mdb.models['abaqusModel8'].parts['Round']
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    p = mdb.models['abaqusModel8'].parts['Round']
    p.seedPart(size=1.4, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['abaqusModel8'].parts['Round']
    p.generateMesh()
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=ON)
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
        engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    a = mdb.models['abaqusModel8'].rootAssembly
    a.regenerate()
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    a = mdb.models['abaqusModel8'].rootAssembly
    p = mdb.models['abaqusModel8'].parts['Round']
    a.Instance(name='Round-1', part=p, dependent=ON)
    a = mdb.models['abaqusModel8'].rootAssembly
    p = mdb.models['abaqusModel8'].parts['Round']
    a.Instance(name='Round-2', part=p, dependent=ON)
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -2.0))
    #: The instance Round-2 was translated by 0., 0., -2. with respect to the assembly coordinate system
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 6.0))
    #: The instance Round-1 was translated by 0., 0., 6. with respect to the assembly coordinate system

    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('PART-1-1', ), vector=(-3.0, -3.0, 0.0))
    #: The instance PART-1-1 was translated by -3., -3., 0. with respect to the assembly coordinate system

    
    a = mdb.models['abaqusModel8'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    mdb.models['abaqusModel8'].ExplicitDynamicsStep(name='Step-1', 
        previous='Initial', timePeriod=3.0, improvedDtMethod=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    mdb.models['abaqusModel8'].steps['Step-1'].setValues(massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 100.0, 0.0, None, 0, 0, 0.0, 0.0, 0, None), ), improvedDtMethod=ON)
    mdb.models['abaqusModel8'].FieldOutputRequest(name='F-Output-2', 
        createStepName='Step-1', variables=('U', 'V', 'A', 'RF', 'RT'))
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
        constraints=ON, connectors=ON, engineeringFeatures=ON, 
        adaptiveMeshConstraints=OFF)
    mdb.models['abaqusModel8'].ContactProperty('IntProp-1')
    mdb.models['abaqusModel8'].interactionProperties['IntProp-1'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
        0.125, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
        fraction=0.005, elasticSlipStiffness=None)
    #: The interaction property "IntProp-1" has been created.
    mdb.models['abaqusModel8'].ContactExp(name='Int-1', createStepName='Step-1')
    mdb.models['abaqusModel8'].interactions['Int-1'].includedPairs.setValuesInStep(
        stepName='Step-1', useAllstar=ON)
    mdb.models['abaqusModel8'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
        stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'), ))
    #: The interaction "Int-1" has been created.
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, interactions=OFF, constraints=OFF, 
        engineeringFeatures=OFF)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=26.3023, 
        farPlane=45.0846, width=21.8334, height=12.6099, cameraPosition=(-31.9957, 
        -4.25204, -12.2387), cameraUpVector=(0.334762, -0.884768, 0.324222), 
        cameraTarget=(-0.693402, -0.101584, 2.66634))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=25.7079, 
        farPlane=45.6825, width=21.34, height=12.325, cameraPosition=(-23.941, 
        -6.20642, -22.7382), cameraUpVector=(0.110802, -0.822844, 0.55736), 
        cameraTarget=(-0.51826, -0.14408, 2.43804))
    a = mdb.models['abaqusModel8'].rootAssembly
    r1 = a.instances['Round-2'].referencePoints
    refPoints1=(r1[3], )
    region = a.Set(referencePoints=refPoints1, name='Set-2')
    mdb.models['abaqusModel8'].DisplacementBC(name='BC-1', createStepName='Step-1', 
        region=region, u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0, 
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
        localCsys=None)
    mdb.models['abaqusModel8'].TabularAmplitude(name='Amp-1', timeSpan=STEP, 
        smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (3.0, 1.0)))
    a = mdb.models['abaqusModel8'].rootAssembly
    r1 = a.instances['Round-1'].referencePoints
    refPoints1=(r1[3], )
    region = a.Set(referencePoints=refPoints1, name='Set-3')
    mdb.models['abaqusModel8'].DisplacementBC(name='BC-2', createStepName='Step-1', 
        region=region, u1=0.0, u2=0.0, u3=-2.0, ur1=0.0, ur2=0.0, ur3=0.0, 
        amplitude='Amp-1', fixed=OFF, distributionType=UNIFORM, fieldName='', 
        localCsys=None)
    mdb.models['abaqusModel8'].fieldOutputRequests['F-Output-1'].setValues(
    numIntervals=40)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=26.1988, 
        farPlane=45.19, width=21.7475, height=12.5603, cameraPosition=(-32.6687, 
        -8.27521, -8.76291), cameraUpVector=(0.578197, -0.815619, -0.0212995), 
        cameraTarget=(-0.70845, -0.189164, 2.74259))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=27.0564, 
        farPlane=44.3306, width=22.4594, height=12.9715, cameraPosition=(-34.7158, 
        -6.60896, -2.01629), cameraUpVector=(0.533339, -0.804698, -0.26079), 
        cameraTarget=(-0.753016, -0.152889, 2.88947))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=27.354, 
        farPlane=44.0263, width=22.7064, height=13.1142, cameraPosition=(-35.095, 
        5.90566, 0.307251), cameraUpVector=(0.18363, -0.975823, -0.118535), 
        cameraTarget=(-0.761262, 0.119269, 2.94))
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
        predefinedFields=OFF, connectors=OFF)
    mdb.Job(name=taskName, model='abaqusModel8', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
        nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
        contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
        resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=6, 
        activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=6)
    mdb.jobs[taskName].submit(consistencyChecking=OFF)

    # Wait for current job completion
    mdb.jobs[taskName].waitForCompletion()
    del mdb.models['abaqusModel8']
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)

    from abaqus import *
    from abaqusConstants import *
    session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=273.328125, 
    height=210.518508911133)
    session.viewports['Viewport: 1'].makeCurrent()
    session.viewports['Viewport: 1'].maximize()
    from caeModules import *
    from driverUtils import executeOnCaeStartup
    executeOnCaeStartup()
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
    Mdb()
    #: A new model database has been created.
    #: The model "Model-1" has been created.
    session.viewports['Viewport: 1'].setValues(displayedObject=None)
    session.viewports['Viewport: 1'].setValues(displayedObject=None)
    odbFileName ='E:/qinyu/daijiabao/suijishengcheng-matlab/'+ taskName + '.odb'
    o1 = session.openOdb(name=odbFileName)
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    session.animationController.setValues(animationType=SCALE_FACTOR, viewports=('Viewport: 1', ))
    session.animationController.play(duration=UNLIMITED)
    session.animationController.animationOptions.setValues(frameRate=28)
    session.animationController.stop()
    odb = session.odbs[odbFileName]
    session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
        NODAL, ((COMPONENT, 'RF3'), )), ('U', NODAL, ((COMPONENT, 'U3'), )), ), 
        nodeSets=("REFERENCE_POINT_ROUND-1      133", ))
    xy1 = session.xyDataObjects['U:U3 PI: ROUND-1 N: 133']
    xy2 = session.xyDataObjects['RF:RF3 PI: ROUND-1 N: 133']
    xy3 = combine(-xy1, -xy2)
    tmpName = xy3.name
    XYPlotName='XYPlot-'+taskName
    xyp = session.XYPlot(XYPlotName)
    chartName = xyp.charts.keys()[0]
    chart = xyp.charts[chartName]
    c1 = session.Curve(xyData=xy3)
    chart.setValues(curvesToPlot=(c1, ), )
    xyp = session.xyPlots[XYPlotName]
    chartName = xyp.charts.keys()[0]
    chart = xyp.charts[chartName]
    x0 = chart.curves[tmpName]
    session.writeXYReport(fileName='E:/qinyu/daijiabao/rpt/' + taskName +'.rpt', xyData=(x0, ))
    session.odbs[odbFileName].close()
# Main Loop

fileList = os.listdir(wsDir)
for file in fileList:
    if file.endswith('.inp'):
        fileName = file[0:-4]
        modelSetUp(wsDir, fileName)