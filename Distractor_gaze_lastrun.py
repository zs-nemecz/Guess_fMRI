#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on május 29, 2025, at 15:10
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'Distractor_gaze'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Asus\\Documents\\pretest_fmri\\Distractor Gaze\\Distractor_gaze_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_dist_instr') is None:
        # initialise key_resp_dist_instr
        key_resp_dist_instr = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_dist_instr',
        )
    if deviceManager.getDevice('key_resp_gaze') is None:
        # initialise key_resp_gaze
        key_resp_gaze = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_gaze',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "distractor_instruction" ---
    dist_instruc = visual.TextStim(win=win, name='dist_instruc',
        text='Es folgt nun eine kurze Aufgabe.\n\nIn der Mitte des Bildschirms erscheint ein Smiley, der nach rechts und links schaut. Neben ihm erscheint ein grüner und roter Kreis.\n\nIhre Aufgabe ist es anzugeben, wo der GRÜNE Kreis auftaucht.\nWenn er LINKS auftaucht drücken Sie TASTE 1, wenn er RECHTS auftaucht drücken Sie TASTE 2. \n\nStarten Sie die Übung durch das Drücken einer beliebigen Taste.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_dist_instr = keyboard.Keyboard(deviceName='key_resp_dist_instr')
    
    # --- Initialize components for Routine "dist_trial" ---
    Smileycentral = visual.ImageStim(
        win=win,
        name='Smileycentral', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_gaze = keyboard.Keyboard(deviceName='key_resp_gaze')
    Smiley = visual.ImageStim(
        win=win,
        name='Smiley', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1,0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    circle = visual.ShapeStim(
        win=win, name='circle',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -0.2157, -1.0000], fillColor=[-1.0000, -0.2157, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    distractorcircle = visual.ShapeStim(
        win=win, name='distractorcircle',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.7255, -0.8431, -0.5294], fillColor=[0.7255, -0.8431, -0.5294],
        opacity=None, depth=-4.0, interpolate=True)
    # Run 'Begin Experiment' code from dist_code
    startTime = core.getTime()
    fback_text = ""
    rt = ""
    
    # --- Initialize components for Routine "feedback" ---
    dist_fback_text = visual.TextStim(win=win, name='dist_fback_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "start_dist" ---
    start_dist_text = visual.TextStim(win=win, name='start_dist_text',
        text='Jetzt beginnt die Aufgabe. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "dist_trial" ---
    Smileycentral = visual.ImageStim(
        win=win,
        name='Smileycentral', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_gaze = keyboard.Keyboard(deviceName='key_resp_gaze')
    Smiley = visual.ImageStim(
        win=win,
        name='Smiley', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1,0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    circle = visual.ShapeStim(
        win=win, name='circle',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -0.2157, -1.0000], fillColor=[-1.0000, -0.2157, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    distractorcircle = visual.ShapeStim(
        win=win, name='distractorcircle',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.7255, -0.8431, -0.5294], fillColor=[0.7255, -0.8431, -0.5294],
        opacity=None, depth=-4.0, interpolate=True)
    # Run 'Begin Experiment' code from dist_code
    startTime = core.getTime()
    fback_text = ""
    rt = ""
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "distractor_instruction" ---
    # create an object to store info about Routine distractor_instruction
    distractor_instruction = data.Routine(
        name='distractor_instruction',
        components=[dist_instruc, key_resp_dist_instr],
    )
    distractor_instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_dist_instr
    key_resp_dist_instr.keys = []
    key_resp_dist_instr.rt = []
    _key_resp_dist_instr_allKeys = []
    # store start times for distractor_instruction
    distractor_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    distractor_instruction.tStart = globalClock.getTime(format='float')
    distractor_instruction.status = STARTED
    thisExp.addData('distractor_instruction.started', distractor_instruction.tStart)
    distractor_instruction.maxDuration = None
    # keep track of which components have finished
    distractor_instructionComponents = distractor_instruction.components
    for thisComponent in distractor_instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "distractor_instruction" ---
    distractor_instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *dist_instruc* updates
        
        # if dist_instruc is starting this frame...
        if dist_instruc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dist_instruc.frameNStart = frameN  # exact frame index
            dist_instruc.tStart = t  # local t and not account for scr refresh
            dist_instruc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dist_instruc, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dist_instruc.started')
            # update status
            dist_instruc.status = STARTED
            dist_instruc.setAutoDraw(True)
        
        # if dist_instruc is active this frame...
        if dist_instruc.status == STARTED:
            # update params
            pass
        
        # *key_resp_dist_instr* updates
        waitOnFlip = False
        
        # if key_resp_dist_instr is starting this frame...
        if key_resp_dist_instr.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            key_resp_dist_instr.frameNStart = frameN  # exact frame index
            key_resp_dist_instr.tStart = t  # local t and not account for scr refresh
            key_resp_dist_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_dist_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_dist_instr.started')
            # update status
            key_resp_dist_instr.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_dist_instr.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_dist_instr.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_dist_instr.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_dist_instr.getKeys(keyList=['2', '3', '4', '5'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_dist_instr_allKeys.extend(theseKeys)
            if len(_key_resp_dist_instr_allKeys):
                key_resp_dist_instr.keys = _key_resp_dist_instr_allKeys[-1].name  # just the last key pressed
                key_resp_dist_instr.rt = _key_resp_dist_instr_allKeys[-1].rt
                key_resp_dist_instr.duration = _key_resp_dist_instr_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            distractor_instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in distractor_instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "distractor_instruction" ---
    for thisComponent in distractor_instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for distractor_instruction
    distractor_instruction.tStop = globalClock.getTime(format='float')
    distractor_instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('distractor_instruction.stopped', distractor_instruction.tStop)
    # check responses
    if key_resp_dist_instr.keys in ['', [], None]:  # No response was made
        key_resp_dist_instr.keys = None
    thisExp.addData('key_resp_dist_instr.keys',key_resp_dist_instr.keys)
    if key_resp_dist_instr.keys != None:  # we had a response
        thisExp.addData('key_resp_dist_instr.rt', key_resp_dist_instr.rt)
        thisExp.addData('key_resp_dist_instr.duration', key_resp_dist_instr.duration)
    thisExp.nextEntry()
    # the Routine "distractor_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    dist_practice = data.TrialHandler2(
        name='dist_practice',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(dist_practice)  # add the loop to the experiment
    thisDist_practice = dist_practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisDist_practice.rgb)
    if thisDist_practice != None:
        for paramName in thisDist_practice:
            globals()[paramName] = thisDist_practice[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisDist_practice in dist_practice:
        currentLoop = dist_practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisDist_practice.rgb)
        if thisDist_practice != None:
            for paramName in thisDist_practice:
                globals()[paramName] = thisDist_practice[paramName]
        
        # --- Prepare to start Routine "dist_trial" ---
        # create an object to store info about Routine dist_trial
        dist_trial = data.Routine(
            name='dist_trial',
            components=[Smileycentral, key_resp_gaze, Smiley, circle, distractorcircle],
        )
        dist_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Smileycentral.setImage('Smiley_central.png')
        # create starting attributes for key_resp_gaze
        key_resp_gaze.keys = []
        key_resp_gaze.rt = []
        _key_resp_gaze_allKeys = []
        Smiley.setImage(Smiley_file)
        circle.setPos(circlePos)
        distractorcircle.setPos(distractorPos)
        # Run 'Begin Routine' code from dist_code
        if core.getTime() - startTime >= 187:
            dist_trials.finished = True
        # store start times for dist_trial
        dist_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dist_trial.tStart = globalClock.getTime(format='float')
        dist_trial.status = STARTED
        thisExp.addData('dist_trial.started', dist_trial.tStart)
        dist_trial.maxDuration = None
        # keep track of which components have finished
        dist_trialComponents = dist_trial.components
        for thisComponent in dist_trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dist_trial" ---
        # if trial has changed, end Routine now
        if isinstance(dist_practice, data.TrialHandler2) and thisDist_practice.thisN != dist_practice.thisTrial.thisN:
            continueRoutine = False
        dist_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Smileycentral* updates
            
            # if Smileycentral is starting this frame...
            if Smileycentral.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Smileycentral.frameNStart = frameN  # exact frame index
                Smileycentral.tStart = t  # local t and not account for scr refresh
                Smileycentral.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Smileycentral, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Smileycentral.started')
                # update status
                Smileycentral.status = STARTED
                Smileycentral.setAutoDraw(True)
            
            # if Smileycentral is active this frame...
            if Smileycentral.status == STARTED:
                # update params
                pass
            
            # if Smileycentral is stopping this frame...
            if Smileycentral.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Smileycentral.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Smileycentral.tStop = t  # not accounting for scr refresh
                    Smileycentral.tStopRefresh = tThisFlipGlobal  # on global time
                    Smileycentral.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Smileycentral.stopped')
                    # update status
                    Smileycentral.status = FINISHED
                    Smileycentral.setAutoDraw(False)
            
            # *key_resp_gaze* updates
            waitOnFlip = False
            
            # if key_resp_gaze is starting this frame...
            if key_resp_gaze.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_gaze.frameNStart = frameN  # exact frame index
                key_resp_gaze.tStart = t  # local t and not account for scr refresh
                key_resp_gaze.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_gaze, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_gaze.started')
                # update status
                key_resp_gaze.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_gaze.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_gaze.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_gaze.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_gaze.getKeys(keyList=['2', '3'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_gaze_allKeys.extend(theseKeys)
                if len(_key_resp_gaze_allKeys):
                    key_resp_gaze.keys = _key_resp_gaze_allKeys[-1].name  # just the last key pressed
                    key_resp_gaze.rt = _key_resp_gaze_allKeys[-1].rt
                    key_resp_gaze.duration = _key_resp_gaze_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_gaze.keys == str(correctResponse)) or (key_resp_gaze.keys == correctResponse):
                        key_resp_gaze.corr = 1
                    else:
                        key_resp_gaze.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *Smiley* updates
            
            # if Smiley is starting this frame...
            if Smiley.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Smiley.frameNStart = frameN  # exact frame index
                Smiley.tStart = t  # local t and not account for scr refresh
                Smiley.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Smiley, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Smiley.started')
                # update status
                Smiley.status = STARTED
                Smiley.setAutoDraw(True)
            
            # if Smiley is active this frame...
            if Smiley.status == STARTED:
                # update params
                pass
            
            # *circle* updates
            
            # if circle is starting this frame...
            if circle.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                circle.frameNStart = frameN  # exact frame index
                circle.tStart = t  # local t and not account for scr refresh
                circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle.started')
                # update status
                circle.status = STARTED
                circle.setAutoDraw(True)
            
            # if circle is active this frame...
            if circle.status == STARTED:
                # update params
                pass
            
            # if circle is stopping this frame...
            if circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > circle.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    circle.tStop = t  # not accounting for scr refresh
                    circle.tStopRefresh = tThisFlipGlobal  # on global time
                    circle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'circle.stopped')
                    # update status
                    circle.status = FINISHED
                    circle.setAutoDraw(False)
            
            # *distractorcircle* updates
            
            # if distractorcircle is starting this frame...
            if distractorcircle.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                distractorcircle.frameNStart = frameN  # exact frame index
                distractorcircle.tStart = t  # local t and not account for scr refresh
                distractorcircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractorcircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'distractorcircle.started')
                # update status
                distractorcircle.status = STARTED
                distractorcircle.setAutoDraw(True)
            
            # if distractorcircle is active this frame...
            if distractorcircle.status == STARTED:
                # update params
                pass
            
            # if distractorcircle is stopping this frame...
            if distractorcircle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractorcircle.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    distractorcircle.tStop = t  # not accounting for scr refresh
                    distractorcircle.tStopRefresh = tThisFlipGlobal  # on global time
                    distractorcircle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'distractorcircle.stopped')
                    # update status
                    distractorcircle.status = FINISHED
                    distractorcircle.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                dist_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dist_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dist_trial" ---
        for thisComponent in dist_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dist_trial
        dist_trial.tStop = globalClock.getTime(format='float')
        dist_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dist_trial.stopped', dist_trial.tStop)
        # check responses
        if key_resp_gaze.keys in ['', [], None]:  # No response was made
            key_resp_gaze.keys = None
            # was no response the correct answer?!
            if str(correctResponse).lower() == 'none':
               key_resp_gaze.corr = 1;  # correct non-response
            else:
               key_resp_gaze.corr = 0;  # failed to respond (incorrectly)
        # store data for dist_practice (TrialHandler)
        dist_practice.addData('key_resp_gaze.keys',key_resp_gaze.keys)
        dist_practice.addData('key_resp_gaze.corr', key_resp_gaze.corr)
        if key_resp_gaze.keys != None:  # we had a response
            dist_practice.addData('key_resp_gaze.rt', key_resp_gaze.rt)
            dist_practice.addData('key_resp_gaze.duration', key_resp_gaze.duration)
        # Run 'End Routine' code from dist_code
        fback_text = ""
        rt = ""
        
        if key_resp_gaze.corr == 1:
            rt = "{:.2f} s".format(key_resp_gaze.rt)
            fback_text = "Gut gemacht!\nReaktionszeit: "
            fback_text = fback_text + rt
        else:
            fback_text = "Sie haben den falschen Kopf gedrückt."
        # the Routine "dist_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[dist_fback_text],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        dist_fback_text.setText(fback_text)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(dist_practice, data.TrialHandler2) and thisDist_practice.thisN != dist_practice.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dist_fback_text* updates
            
            # if dist_fback_text is starting this frame...
            if dist_fback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_fback_text.frameNStart = frameN  # exact frame index
                dist_fback_text.tStart = t  # local t and not account for scr refresh
                dist_fback_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_fback_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_fback_text.started')
                # update status
                dist_fback_text.status = STARTED
                dist_fback_text.setAutoDraw(True)
            
            # if dist_fback_text is active this frame...
            if dist_fback_text.status == STARTED:
                # update params
                pass
            
            # if dist_fback_text is stopping this frame...
            if dist_fback_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dist_fback_text.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dist_fback_text.tStop = t  # not accounting for scr refresh
                    dist_fback_text.tStopRefresh = tThisFlipGlobal  # on global time
                    dist_fback_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dist_fback_text.stopped')
                    # update status
                    dist_fback_text.status = FINISHED
                    dist_fback_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'dist_practice'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "start_dist" ---
    # create an object to store info about Routine start_dist
    start_dist = data.Routine(
        name='start_dist',
        components=[start_dist_text],
    )
    start_dist.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for start_dist
    start_dist.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start_dist.tStart = globalClock.getTime(format='float')
    start_dist.status = STARTED
    thisExp.addData('start_dist.started', start_dist.tStart)
    start_dist.maxDuration = None
    # keep track of which components have finished
    start_distComponents = start_dist.components
    for thisComponent in start_dist.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_dist" ---
    start_dist.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_dist_text* updates
        
        # if start_dist_text is starting this frame...
        if start_dist_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_dist_text.frameNStart = frameN  # exact frame index
            start_dist_text.tStart = t  # local t and not account for scr refresh
            start_dist_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_dist_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_dist_text.started')
            # update status
            start_dist_text.status = STARTED
            start_dist_text.setAutoDraw(True)
        
        # if start_dist_text is active this frame...
        if start_dist_text.status == STARTED:
            # update params
            pass
        
        # if start_dist_text is stopping this frame...
        if start_dist_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > start_dist_text.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                start_dist_text.tStop = t  # not accounting for scr refresh
                start_dist_text.tStopRefresh = tThisFlipGlobal  # on global time
                start_dist_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_dist_text.stopped')
                # update status
                start_dist_text.status = FINISHED
                start_dist_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start_dist.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_dist.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_dist" ---
    for thisComponent in start_dist.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start_dist
    start_dist.tStop = globalClock.getTime(format='float')
    start_dist.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start_dist.stopped', start_dist.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if start_dist.maxDurationReached:
        routineTimer.addTime(-start_dist.maxDuration)
    elif start_dist.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    dist_trials = data.TrialHandler2(
        name='dist_trials',
        nReps=1999.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(dist_trials)  # add the loop to the experiment
    thisDist_trial = dist_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisDist_trial.rgb)
    if thisDist_trial != None:
        for paramName in thisDist_trial:
            globals()[paramName] = thisDist_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisDist_trial in dist_trials:
        currentLoop = dist_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisDist_trial.rgb)
        if thisDist_trial != None:
            for paramName in thisDist_trial:
                globals()[paramName] = thisDist_trial[paramName]
        
        # --- Prepare to start Routine "dist_trial" ---
        # create an object to store info about Routine dist_trial
        dist_trial = data.Routine(
            name='dist_trial',
            components=[Smileycentral, key_resp_gaze, Smiley, circle, distractorcircle],
        )
        dist_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Smileycentral.setImage('Smiley_central.png')
        # create starting attributes for key_resp_gaze
        key_resp_gaze.keys = []
        key_resp_gaze.rt = []
        _key_resp_gaze_allKeys = []
        Smiley.setImage(Smiley_file)
        circle.setPos(circlePos)
        distractorcircle.setPos(distractorPos)
        # Run 'Begin Routine' code from dist_code
        if core.getTime() - startTime >= 187:
            dist_trials.finished = True
        # store start times for dist_trial
        dist_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dist_trial.tStart = globalClock.getTime(format='float')
        dist_trial.status = STARTED
        thisExp.addData('dist_trial.started', dist_trial.tStart)
        dist_trial.maxDuration = None
        # keep track of which components have finished
        dist_trialComponents = dist_trial.components
        for thisComponent in dist_trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dist_trial" ---
        # if trial has changed, end Routine now
        if isinstance(dist_trials, data.TrialHandler2) and thisDist_trial.thisN != dist_trials.thisTrial.thisN:
            continueRoutine = False
        dist_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Smileycentral* updates
            
            # if Smileycentral is starting this frame...
            if Smileycentral.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Smileycentral.frameNStart = frameN  # exact frame index
                Smileycentral.tStart = t  # local t and not account for scr refresh
                Smileycentral.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Smileycentral, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Smileycentral.started')
                # update status
                Smileycentral.status = STARTED
                Smileycentral.setAutoDraw(True)
            
            # if Smileycentral is active this frame...
            if Smileycentral.status == STARTED:
                # update params
                pass
            
            # if Smileycentral is stopping this frame...
            if Smileycentral.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Smileycentral.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Smileycentral.tStop = t  # not accounting for scr refresh
                    Smileycentral.tStopRefresh = tThisFlipGlobal  # on global time
                    Smileycentral.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Smileycentral.stopped')
                    # update status
                    Smileycentral.status = FINISHED
                    Smileycentral.setAutoDraw(False)
            
            # *key_resp_gaze* updates
            waitOnFlip = False
            
            # if key_resp_gaze is starting this frame...
            if key_resp_gaze.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_gaze.frameNStart = frameN  # exact frame index
                key_resp_gaze.tStart = t  # local t and not account for scr refresh
                key_resp_gaze.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_gaze, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_gaze.started')
                # update status
                key_resp_gaze.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_gaze.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_gaze.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_gaze.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_gaze.getKeys(keyList=['2', '3'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_gaze_allKeys.extend(theseKeys)
                if len(_key_resp_gaze_allKeys):
                    key_resp_gaze.keys = _key_resp_gaze_allKeys[-1].name  # just the last key pressed
                    key_resp_gaze.rt = _key_resp_gaze_allKeys[-1].rt
                    key_resp_gaze.duration = _key_resp_gaze_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_gaze.keys == str(correctResponse)) or (key_resp_gaze.keys == correctResponse):
                        key_resp_gaze.corr = 1
                    else:
                        key_resp_gaze.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *Smiley* updates
            
            # if Smiley is starting this frame...
            if Smiley.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Smiley.frameNStart = frameN  # exact frame index
                Smiley.tStart = t  # local t and not account for scr refresh
                Smiley.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Smiley, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Smiley.started')
                # update status
                Smiley.status = STARTED
                Smiley.setAutoDraw(True)
            
            # if Smiley is active this frame...
            if Smiley.status == STARTED:
                # update params
                pass
            
            # *circle* updates
            
            # if circle is starting this frame...
            if circle.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                circle.frameNStart = frameN  # exact frame index
                circle.tStart = t  # local t and not account for scr refresh
                circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle.started')
                # update status
                circle.status = STARTED
                circle.setAutoDraw(True)
            
            # if circle is active this frame...
            if circle.status == STARTED:
                # update params
                pass
            
            # if circle is stopping this frame...
            if circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > circle.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    circle.tStop = t  # not accounting for scr refresh
                    circle.tStopRefresh = tThisFlipGlobal  # on global time
                    circle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'circle.stopped')
                    # update status
                    circle.status = FINISHED
                    circle.setAutoDraw(False)
            
            # *distractorcircle* updates
            
            # if distractorcircle is starting this frame...
            if distractorcircle.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                distractorcircle.frameNStart = frameN  # exact frame index
                distractorcircle.tStart = t  # local t and not account for scr refresh
                distractorcircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractorcircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'distractorcircle.started')
                # update status
                distractorcircle.status = STARTED
                distractorcircle.setAutoDraw(True)
            
            # if distractorcircle is active this frame...
            if distractorcircle.status == STARTED:
                # update params
                pass
            
            # if distractorcircle is stopping this frame...
            if distractorcircle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractorcircle.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    distractorcircle.tStop = t  # not accounting for scr refresh
                    distractorcircle.tStopRefresh = tThisFlipGlobal  # on global time
                    distractorcircle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'distractorcircle.stopped')
                    # update status
                    distractorcircle.status = FINISHED
                    distractorcircle.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                dist_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dist_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dist_trial" ---
        for thisComponent in dist_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dist_trial
        dist_trial.tStop = globalClock.getTime(format='float')
        dist_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dist_trial.stopped', dist_trial.tStop)
        # check responses
        if key_resp_gaze.keys in ['', [], None]:  # No response was made
            key_resp_gaze.keys = None
            # was no response the correct answer?!
            if str(correctResponse).lower() == 'none':
               key_resp_gaze.corr = 1;  # correct non-response
            else:
               key_resp_gaze.corr = 0;  # failed to respond (incorrectly)
        # store data for dist_trials (TrialHandler)
        dist_trials.addData('key_resp_gaze.keys',key_resp_gaze.keys)
        dist_trials.addData('key_resp_gaze.corr', key_resp_gaze.corr)
        if key_resp_gaze.keys != None:  # we had a response
            dist_trials.addData('key_resp_gaze.rt', key_resp_gaze.rt)
            dist_trials.addData('key_resp_gaze.duration', key_resp_gaze.duration)
        # Run 'End Routine' code from dist_code
        fback_text = ""
        rt = ""
        
        if key_resp_gaze.corr == 1:
            rt = "{:.2f} s".format(key_resp_gaze.rt)
            fback_text = "Gut gemacht!\nReaktionszeit: "
            fback_text = fback_text + rt
        else:
            fback_text = "Sie haben den falschen Kopf gedrückt."
        # the Routine "dist_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1999.0 repeats of 'dist_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
