﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on április 01, 2025, at 18:15
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
expName = 'guess'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'ID-Nummer': f"{randint(0, 999999):06.0f}",
    'Alter': '',
    'Geschlecht (optional)': '',
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
_winSize = [1440, 900]
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
    filename = u'data/%s_%s_%s' % (expInfo['ID-Nummer'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Asus\\Documents\\pretest_fmri\\Guess_fMRI\\part3_lastrun.py',
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
    if deviceManager.getDevice('sm_instructions1_key') is None:
        # initialise sm_instructions1_key
        sm_instructions1_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='sm_instructions1_key',
        )
    if deviceManager.getDevice('sm_instructions2_key') is None:
        # initialise sm_instructions2_key
        sm_instructions2_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='sm_instructions2_key',
        )
    if deviceManager.getDevice('living_nonliving') is None:
        # initialise living_nonliving
        living_nonliving = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='living_nonliving',
        )
    if deviceManager.getDevice('end_mapping') is None:
        # initialise end_mapping
        end_mapping = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_mapping',
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
    
    # --- Initialize components for Routine "welcome" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='Willkommen zum Ratespiel Experiment!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from experiment_setup
    # import random for letter randomization
    import random as rnd
    
    # --- Initialize components for Routine "instructions_semantic_mapping1" ---
    sm_instructions1_text = visual.TextStim(win=win, name='sm_instructions1_text',
        text='Im letzten Teil des Experiments wird Ihnen eine Liste von Wörtern jeweils einzeln auf dem Bildschirm präsentiert. Sie müssen entscheiden, ob das Wort etwas Lebendiges oder Nicht-Lebendiges bezeichnet.\n\nDrücken Sie die LEERTASTE, um fortzufahren.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sm_instructions1_key = keyboard.Keyboard(deviceName='sm_instructions1_key')
    
    # --- Initialize components for Routine "instructions_semantic_mapping2" ---
    sm_instructions2_text = visual.TextStim(win=win, name='sm_instructions2_text',
        text='Bei manchen Wörtern ist es schwieriger, diese Entscheidung zu treffen als bei anderen. Es gibt nicht immer eine richtige oder falsche Antwort. Verlassen Sie sich auf Ihre Intuition, die Entscheidung liegt bei Ihnen!\n\nGeben Sie Ihre Antwort mit den Tasten F / G an.\n\nDrücken Sie die LEERTASTE, um die Aufgabe zu starten.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sm_instructions2_key = keyboard.Keyboard(deviceName='sm_instructions2_key')
    
    # --- Initialize components for Routine "load_iti" ---
    
    # --- Initialize components for Routine "reset_run" ---
    
    # --- Initialize components for Routine "shuffle_items" ---
    
    # --- Initialize components for Routine "set_up_runs" ---
    # Run 'Begin Experiment' code from run_setup
    my_runs = []
    
    # --- Initialize components for Routine "iti" ---
    iti_blank = visual.TextStim(win=win, name='iti_blank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "item" ---
    # Run 'Begin Experiment' code from set_item
    list_item = "None"
    living_nonliving = keyboard.Keyboard(deviceName='living_nonliving')
    end_mapping = keyboard.Keyboard(deviceName='end_mapping')
    living_nonliving_text = visual.TextStim(win=win, name='living_nonliving_text',
        text='F) Lebendig        G) Nicht lebendig',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    semantic_map_item = visual.TextStim(win=win, name='semantic_map_item',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "thanks" ---
    thanks_text = visual.TextStim(win=win, name='thanks_text',
        text='Das ist das Ende des Experiments.\n\nVielen Dank!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
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
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[welcome_text],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
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
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # if welcome_text is stopping this frame...
        if welcome_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > welcome_text.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                welcome_text.tStop = t  # not accounting for scr refresh
                welcome_text.tStopRefresh = tThisFlipGlobal  # on global time
                welcome_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'welcome_text.stopped')
                # update status
                welcome_text.status = FINISHED
                welcome_text.setAutoDraw(False)
        
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
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if welcome.maxDurationReached:
        routineTimer.addTime(-welcome.maxDuration)
    elif welcome.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instructions_semantic_mapping1" ---
    # create an object to store info about Routine instructions_semantic_mapping1
    instructions_semantic_mapping1 = data.Routine(
        name='instructions_semantic_mapping1',
        components=[sm_instructions1_text, sm_instructions1_key],
    )
    instructions_semantic_mapping1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for sm_instructions1_key
    sm_instructions1_key.keys = []
    sm_instructions1_key.rt = []
    _sm_instructions1_key_allKeys = []
    # store start times for instructions_semantic_mapping1
    instructions_semantic_mapping1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_semantic_mapping1.tStart = globalClock.getTime(format='float')
    instructions_semantic_mapping1.status = STARTED
    thisExp.addData('instructions_semantic_mapping1.started', instructions_semantic_mapping1.tStart)
    instructions_semantic_mapping1.maxDuration = None
    # keep track of which components have finished
    instructions_semantic_mapping1Components = instructions_semantic_mapping1.components
    for thisComponent in instructions_semantic_mapping1.components:
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
    
    # --- Run Routine "instructions_semantic_mapping1" ---
    instructions_semantic_mapping1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sm_instructions1_text* updates
        
        # if sm_instructions1_text is starting this frame...
        if sm_instructions1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sm_instructions1_text.frameNStart = frameN  # exact frame index
            sm_instructions1_text.tStart = t  # local t and not account for scr refresh
            sm_instructions1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sm_instructions1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sm_instructions1_text.started')
            # update status
            sm_instructions1_text.status = STARTED
            sm_instructions1_text.setAutoDraw(True)
        
        # if sm_instructions1_text is active this frame...
        if sm_instructions1_text.status == STARTED:
            # update params
            pass
        
        # *sm_instructions1_key* updates
        waitOnFlip = False
        
        # if sm_instructions1_key is starting this frame...
        if sm_instructions1_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sm_instructions1_key.frameNStart = frameN  # exact frame index
            sm_instructions1_key.tStart = t  # local t and not account for scr refresh
            sm_instructions1_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sm_instructions1_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sm_instructions1_key.started')
            # update status
            sm_instructions1_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(sm_instructions1_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(sm_instructions1_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if sm_instructions1_key.status == STARTED and not waitOnFlip:
            theseKeys = sm_instructions1_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _sm_instructions1_key_allKeys.extend(theseKeys)
            if len(_sm_instructions1_key_allKeys):
                sm_instructions1_key.keys = _sm_instructions1_key_allKeys[-1].name  # just the last key pressed
                sm_instructions1_key.rt = _sm_instructions1_key_allKeys[-1].rt
                sm_instructions1_key.duration = _sm_instructions1_key_allKeys[-1].duration
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
            instructions_semantic_mapping1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_semantic_mapping1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_semantic_mapping1" ---
    for thisComponent in instructions_semantic_mapping1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_semantic_mapping1
    instructions_semantic_mapping1.tStop = globalClock.getTime(format='float')
    instructions_semantic_mapping1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_semantic_mapping1.stopped', instructions_semantic_mapping1.tStop)
    # check responses
    if sm_instructions1_key.keys in ['', [], None]:  # No response was made
        sm_instructions1_key.keys = None
    thisExp.addData('sm_instructions1_key.keys',sm_instructions1_key.keys)
    if sm_instructions1_key.keys != None:  # we had a response
        thisExp.addData('sm_instructions1_key.rt', sm_instructions1_key.rt)
        thisExp.addData('sm_instructions1_key.duration', sm_instructions1_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions_semantic_mapping1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_semantic_mapping2" ---
    # create an object to store info about Routine instructions_semantic_mapping2
    instructions_semantic_mapping2 = data.Routine(
        name='instructions_semantic_mapping2',
        components=[sm_instructions2_text, sm_instructions2_key],
    )
    instructions_semantic_mapping2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for sm_instructions2_key
    sm_instructions2_key.keys = []
    sm_instructions2_key.rt = []
    _sm_instructions2_key_allKeys = []
    # store start times for instructions_semantic_mapping2
    instructions_semantic_mapping2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_semantic_mapping2.tStart = globalClock.getTime(format='float')
    instructions_semantic_mapping2.status = STARTED
    thisExp.addData('instructions_semantic_mapping2.started', instructions_semantic_mapping2.tStart)
    instructions_semantic_mapping2.maxDuration = None
    # keep track of which components have finished
    instructions_semantic_mapping2Components = instructions_semantic_mapping2.components
    for thisComponent in instructions_semantic_mapping2.components:
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
    
    # --- Run Routine "instructions_semantic_mapping2" ---
    instructions_semantic_mapping2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sm_instructions2_text* updates
        
        # if sm_instructions2_text is starting this frame...
        if sm_instructions2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sm_instructions2_text.frameNStart = frameN  # exact frame index
            sm_instructions2_text.tStart = t  # local t and not account for scr refresh
            sm_instructions2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sm_instructions2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sm_instructions2_text.started')
            # update status
            sm_instructions2_text.status = STARTED
            sm_instructions2_text.setAutoDraw(True)
        
        # if sm_instructions2_text is active this frame...
        if sm_instructions2_text.status == STARTED:
            # update params
            pass
        
        # *sm_instructions2_key* updates
        waitOnFlip = False
        
        # if sm_instructions2_key is starting this frame...
        if sm_instructions2_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sm_instructions2_key.frameNStart = frameN  # exact frame index
            sm_instructions2_key.tStart = t  # local t and not account for scr refresh
            sm_instructions2_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sm_instructions2_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sm_instructions2_key.started')
            # update status
            sm_instructions2_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(sm_instructions2_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(sm_instructions2_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if sm_instructions2_key.status == STARTED and not waitOnFlip:
            theseKeys = sm_instructions2_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _sm_instructions2_key_allKeys.extend(theseKeys)
            if len(_sm_instructions2_key_allKeys):
                sm_instructions2_key.keys = _sm_instructions2_key_allKeys[-1].name  # just the last key pressed
                sm_instructions2_key.rt = _sm_instructions2_key_allKeys[-1].rt
                sm_instructions2_key.duration = _sm_instructions2_key_allKeys[-1].duration
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
            instructions_semantic_mapping2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_semantic_mapping2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_semantic_mapping2" ---
    for thisComponent in instructions_semantic_mapping2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_semantic_mapping2
    instructions_semantic_mapping2.tStop = globalClock.getTime(format='float')
    instructions_semantic_mapping2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_semantic_mapping2.stopped', instructions_semantic_mapping2.tStop)
    # check responses
    if sm_instructions2_key.keys in ['', [], None]:  # No response was made
        sm_instructions2_key.keys = None
    thisExp.addData('sm_instructions2_key.keys',sm_instructions2_key.keys)
    if sm_instructions2_key.keys != None:  # we had a response
        thisExp.addData('sm_instructions2_key.rt', sm_instructions2_key.rt)
        thisExp.addData('sm_instructions2_key.duration', sm_instructions2_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions_semantic_mapping2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # set up handler to look after randomisation of conditions etc
        sm_itis = data.TrialHandler2(
            name='sm_itis',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('sm_iti_randomization.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(sm_itis)  # add the loop to the experiment
        thisSm_iti = sm_itis.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisSm_iti.rgb)
        if thisSm_iti != None:
            for paramName in thisSm_iti:
                globals()[paramName] = thisSm_iti[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisSm_iti in sm_itis:
            currentLoop = sm_itis
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisSm_iti.rgb)
            if thisSm_iti != None:
                for paramName in thisSm_iti:
                    globals()[paramName] = thisSm_iti[paramName]
            
            # --- Prepare to start Routine "load_iti" ---
            # create an object to store info about Routine load_iti
            load_iti = data.Routine(
                name='load_iti',
                components=[],
            )
            load_iti.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for load_iti
            load_iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            load_iti.tStart = globalClock.getTime(format='float')
            load_iti.status = STARTED
            thisExp.addData('load_iti.started', load_iti.tStart)
            load_iti.maxDuration = None
            # keep track of which components have finished
            load_itiComponents = load_iti.components
            for thisComponent in load_iti.components:
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
            
            # --- Run Routine "load_iti" ---
            # if trial has changed, end Routine now
            if isinstance(sm_itis, data.TrialHandler2) and thisSm_iti.thisN != sm_itis.thisTrial.thisN:
                continueRoutine = False
            load_iti.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                    load_iti.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in load_iti.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "load_iti" ---
            for thisComponent in load_iti.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for load_iti
            load_iti.tStop = globalClock.getTime(format='float')
            load_iti.tStopRefresh = tThisFlipGlobal
            thisExp.addData('load_iti.stopped', load_iti.tStop)
            # the Routine "load_iti" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'sm_itis'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "reset_run" ---
        # create an object to store info about Routine reset_run
        reset_run = data.Routine(
            name='reset_run',
            components=[],
        )
        reset_run.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from create_runs
        run1 = []
        run2 = []
        run3 = []
        run4 = []
        # store start times for reset_run
        reset_run.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        reset_run.tStart = globalClock.getTime(format='float')
        reset_run.status = STARTED
        thisExp.addData('reset_run.started', reset_run.tStart)
        reset_run.maxDuration = None
        # keep track of which components have finished
        reset_runComponents = reset_run.components
        for thisComponent in reset_run.components:
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
        
        # --- Run Routine "reset_run" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        reset_run.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
                reset_run.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reset_run.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reset_run" ---
        for thisComponent in reset_run.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for reset_run
        reset_run.tStop = globalClock.getTime(format='float')
        reset_run.tStopRefresh = tThisFlipGlobal
        thisExp.addData('reset_run.stopped', reset_run.tStop)
        # the Routine "reset_run" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        load_and_shuffle = data.TrialHandler2(
            name='load_and_shuffle',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('sm_items.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(load_and_shuffle)  # add the loop to the experiment
        thisLoad_and_shuffle = load_and_shuffle.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoad_and_shuffle.rgb)
        if thisLoad_and_shuffle != None:
            for paramName in thisLoad_and_shuffle:
                globals()[paramName] = thisLoad_and_shuffle[paramName]
        
        for thisLoad_and_shuffle in load_and_shuffle:
            currentLoop = load_and_shuffle
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisLoad_and_shuffle.rgb)
            if thisLoad_and_shuffle != None:
                for paramName in thisLoad_and_shuffle:
                    globals()[paramName] = thisLoad_and_shuffle[paramName]
            
            # --- Prepare to start Routine "shuffle_items" ---
            # create an object to store info about Routine shuffle_items
            shuffle_items = data.Routine(
                name='shuffle_items',
                components=[],
            )
            shuffle_items.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from shuffle_lists
            these_items = [Cue, Target, Mediator1, Mediator2] # assign elements from this row
            rnd.shuffle(these_items) # shuffle order
            
            # append to separate runs
            run1.append(these_items[0])
            run2.append(these_items[1])
            run3.append(these_items[2])
            run4.append(these_items[3])
            
            # store start times for shuffle_items
            shuffle_items.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            shuffle_items.tStart = globalClock.getTime(format='float')
            shuffle_items.status = STARTED
            thisExp.addData('shuffle_items.started', shuffle_items.tStart)
            shuffle_items.maxDuration = None
            # keep track of which components have finished
            shuffle_itemsComponents = shuffle_items.components
            for thisComponent in shuffle_items.components:
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
            
            # --- Run Routine "shuffle_items" ---
            # if trial has changed, end Routine now
            if isinstance(load_and_shuffle, data.TrialHandler2) and thisLoad_and_shuffle.thisN != load_and_shuffle.thisTrial.thisN:
                continueRoutine = False
            shuffle_items.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                    shuffle_items.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in shuffle_items.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "shuffle_items" ---
            for thisComponent in shuffle_items.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for shuffle_items
            shuffle_items.tStop = globalClock.getTime(format='float')
            shuffle_items.tStopRefresh = tThisFlipGlobal
            thisExp.addData('shuffle_items.stopped', shuffle_items.tStop)
            # the Routine "shuffle_items" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed 1.0 repeats of 'load_and_shuffle'
        
        
        # --- Prepare to start Routine "set_up_runs" ---
        # create an object to store info about Routine set_up_runs
        set_up_runs = data.Routine(
            name='set_up_runs',
            components=[],
        )
        set_up_runs.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from run_setup
        print(len(run1))
        print(len(run2))
        print(len(run3))
        print(len(run4))
        
        rnd.shuffle(run1)
        rnd.shuffle(run2)
        rnd.shuffle(run3)
        rnd.shuffle(run4)
        
        my_runs = [run1, run2, run3, run4]
        
        print(my_runs)
        
        # store start times for set_up_runs
        set_up_runs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        set_up_runs.tStart = globalClock.getTime(format='float')
        set_up_runs.status = STARTED
        thisExp.addData('set_up_runs.started', set_up_runs.tStart)
        set_up_runs.maxDuration = None
        # keep track of which components have finished
        set_up_runsComponents = set_up_runs.components
        for thisComponent in set_up_runs.components:
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
        
        # --- Run Routine "set_up_runs" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        set_up_runs.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
                set_up_runs.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in set_up_runs.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "set_up_runs" ---
        for thisComponent in set_up_runs.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for set_up_runs
        set_up_runs.tStop = globalClock.getTime(format='float')
        set_up_runs.tStopRefresh = tThisFlipGlobal
        thisExp.addData('set_up_runs.stopped', set_up_runs.tStop)
        # the Routine "set_up_runs" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        runs = data.TrialHandler2(
            name='runs',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('run_sheets.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(runs)  # add the loop to the experiment
        thisRun = runs.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun:
                globals()[paramName] = thisRun[paramName]
        
        for thisRun in runs:
            currentLoop = runs
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
            if thisRun != None:
                for paramName in thisRun:
                    globals()[paramName] = thisRun[paramName]
            
            # set up handler to look after randomisation of conditions etc
            semantic_mapping_run = data.TrialHandler2(
                name='semantic_mapping_run',
                nReps=1.0, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(this_sheet), 
                seed=None, 
            )
            thisExp.addLoop(semantic_mapping_run)  # add the loop to the experiment
            thisSemantic_mapping_run = semantic_mapping_run.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisSemantic_mapping_run.rgb)
            if thisSemantic_mapping_run != None:
                for paramName in thisSemantic_mapping_run:
                    globals()[paramName] = thisSemantic_mapping_run[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisSemantic_mapping_run in semantic_mapping_run:
                currentLoop = semantic_mapping_run
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisSemantic_mapping_run.rgb)
                if thisSemantic_mapping_run != None:
                    for paramName in thisSemantic_mapping_run:
                        globals()[paramName] = thisSemantic_mapping_run[paramName]
                
                # --- Prepare to start Routine "iti" ---
                # create an object to store info about Routine iti
                iti = data.Routine(
                    name='iti',
                    components=[iti_blank],
                )
                iti.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # store start times for iti
                iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                iti.tStart = globalClock.getTime(format='float')
                iti.status = STARTED
                thisExp.addData('iti.started', iti.tStart)
                iti.maxDuration = None
                # keep track of which components have finished
                itiComponents = iti.components
                for thisComponent in iti.components:
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
                
                # --- Run Routine "iti" ---
                # if trial has changed, end Routine now
                if isinstance(semantic_mapping_run, data.TrialHandler2) and thisSemantic_mapping_run.thisN != semantic_mapping_run.thisTrial.thisN:
                    continueRoutine = False
                iti.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 4.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *iti_blank* updates
                    
                    # if iti_blank is starting this frame...
                    if iti_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        iti_blank.frameNStart = frameN  # exact frame index
                        iti_blank.tStart = t  # local t and not account for scr refresh
                        iti_blank.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(iti_blank, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'iti_blank.started')
                        # update status
                        iti_blank.status = STARTED
                        iti_blank.setAutoDraw(True)
                    
                    # if iti_blank is active this frame...
                    if iti_blank.status == STARTED:
                        # update params
                        pass
                    
                    # if iti_blank is stopping this frame...
                    if iti_blank.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > iti_blank.tStartRefresh + 4.0-frameTolerance:
                            # keep track of stop time/frame for later
                            iti_blank.tStop = t  # not accounting for scr refresh
                            iti_blank.tStopRefresh = tThisFlipGlobal  # on global time
                            iti_blank.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'iti_blank.stopped')
                            # update status
                            iti_blank.status = FINISHED
                            iti_blank.setAutoDraw(False)
                    
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
                        iti.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in iti.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "iti" ---
                for thisComponent in iti.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for iti
                iti.tStop = globalClock.getTime(format='float')
                iti.tStopRefresh = tThisFlipGlobal
                thisExp.addData('iti.stopped', iti.tStop)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if iti.maxDurationReached:
                    routineTimer.addTime(-iti.maxDuration)
                elif iti.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-4.000000)
                
                # --- Prepare to start Routine "item" ---
                # create an object to store info about Routine item
                item = data.Routine(
                    name='item',
                    components=[living_nonliving, end_mapping, living_nonliving_text, semantic_map_item],
                )
                item.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from set_item
                this_list = int(thisList)
                this_run = my_runs[this_list]
                this_item = int(thisIndex)
                list_item = this_run[this_item]
                # create starting attributes for living_nonliving
                living_nonliving.keys = []
                living_nonliving.rt = []
                _living_nonliving_allKeys = []
                # create starting attributes for end_mapping
                end_mapping.keys = []
                end_mapping.rt = []
                _end_mapping_allKeys = []
                semantic_map_item.setText(list_item)
                # store start times for item
                item.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                item.tStart = globalClock.getTime(format='float')
                item.status = STARTED
                thisExp.addData('item.started', item.tStart)
                item.maxDuration = None
                # keep track of which components have finished
                itemComponents = item.components
                for thisComponent in item.components:
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
                
                # --- Run Routine "item" ---
                # if trial has changed, end Routine now
                if isinstance(semantic_mapping_run, data.TrialHandler2) and thisSemantic_mapping_run.thisN != semantic_mapping_run.thisTrial.thisN:
                    continueRoutine = False
                item.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.5:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *living_nonliving* updates
                    waitOnFlip = False
                    
                    # if living_nonliving is starting this frame...
                    if living_nonliving.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        living_nonliving.frameNStart = frameN  # exact frame index
                        living_nonliving.tStart = t  # local t and not account for scr refresh
                        living_nonliving.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(living_nonliving, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'living_nonliving.started')
                        # update status
                        living_nonliving.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(living_nonliving.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(living_nonliving.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if living_nonliving is stopping this frame...
                    if living_nonliving.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > living_nonliving.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            living_nonliving.tStop = t  # not accounting for scr refresh
                            living_nonliving.tStopRefresh = tThisFlipGlobal  # on global time
                            living_nonliving.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'living_nonliving.stopped')
                            # update status
                            living_nonliving.status = FINISHED
                            living_nonliving.status = FINISHED
                    if living_nonliving.status == STARTED and not waitOnFlip:
                        theseKeys = living_nonliving.getKeys(keyList=['f','g'], ignoreKeys=["escape"], waitRelease=False)
                        _living_nonliving_allKeys.extend(theseKeys)
                        if len(_living_nonliving_allKeys):
                            living_nonliving.keys = _living_nonliving_allKeys[-1].name  # just the last key pressed
                            living_nonliving.rt = _living_nonliving_allKeys[-1].rt
                            living_nonliving.duration = _living_nonliving_allKeys[-1].duration
                    
                    # *end_mapping* updates
                    waitOnFlip = False
                    
                    # if end_mapping is starting this frame...
                    if end_mapping.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        end_mapping.frameNStart = frameN  # exact frame index
                        end_mapping.tStart = t  # local t and not account for scr refresh
                        end_mapping.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(end_mapping, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'end_mapping.started')
                        # update status
                        end_mapping.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(end_mapping.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(end_mapping.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if end_mapping is stopping this frame...
                    if end_mapping.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > end_mapping.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            end_mapping.tStop = t  # not accounting for scr refresh
                            end_mapping.tStopRefresh = tThisFlipGlobal  # on global time
                            end_mapping.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'end_mapping.stopped')
                            # update status
                            end_mapping.status = FINISHED
                            end_mapping.status = FINISHED
                    if end_mapping.status == STARTED and not waitOnFlip:
                        theseKeys = end_mapping.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _end_mapping_allKeys.extend(theseKeys)
                        if len(_end_mapping_allKeys):
                            end_mapping.keys = _end_mapping_allKeys[-1].name  # just the last key pressed
                            end_mapping.rt = _end_mapping_allKeys[-1].rt
                            end_mapping.duration = _end_mapping_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *living_nonliving_text* updates
                    
                    # if living_nonliving_text is starting this frame...
                    if living_nonliving_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        living_nonliving_text.frameNStart = frameN  # exact frame index
                        living_nonliving_text.tStart = t  # local t and not account for scr refresh
                        living_nonliving_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(living_nonliving_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'living_nonliving_text.started')
                        # update status
                        living_nonliving_text.status = STARTED
                        living_nonliving_text.setAutoDraw(True)
                    
                    # if living_nonliving_text is active this frame...
                    if living_nonliving_text.status == STARTED:
                        # update params
                        pass
                    
                    # if living_nonliving_text is stopping this frame...
                    if living_nonliving_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > living_nonliving_text.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            living_nonliving_text.tStop = t  # not accounting for scr refresh
                            living_nonliving_text.tStopRefresh = tThisFlipGlobal  # on global time
                            living_nonliving_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'living_nonliving_text.stopped')
                            # update status
                            living_nonliving_text.status = FINISHED
                            living_nonliving_text.setAutoDraw(False)
                    
                    # *semantic_map_item* updates
                    
                    # if semantic_map_item is starting this frame...
                    if semantic_map_item.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        semantic_map_item.frameNStart = frameN  # exact frame index
                        semantic_map_item.tStart = t  # local t and not account for scr refresh
                        semantic_map_item.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(semantic_map_item, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'semantic_map_item.started')
                        # update status
                        semantic_map_item.status = STARTED
                        semantic_map_item.setAutoDraw(True)
                    
                    # if semantic_map_item is active this frame...
                    if semantic_map_item.status == STARTED:
                        # update params
                        pass
                    
                    # if semantic_map_item is stopping this frame...
                    if semantic_map_item.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > semantic_map_item.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            semantic_map_item.tStop = t  # not accounting for scr refresh
                            semantic_map_item.tStopRefresh = tThisFlipGlobal  # on global time
                            semantic_map_item.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'semantic_map_item.stopped')
                            # update status
                            semantic_map_item.status = FINISHED
                            semantic_map_item.setAutoDraw(False)
                    
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
                        item.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in item.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "item" ---
                for thisComponent in item.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for item
                item.tStop = globalClock.getTime(format='float')
                item.tStopRefresh = tThisFlipGlobal
                thisExp.addData('item.stopped', item.tStop)
                # check responses
                if living_nonliving.keys in ['', [], None]:  # No response was made
                    living_nonliving.keys = None
                semantic_mapping_run.addData('living_nonliving.keys',living_nonliving.keys)
                if living_nonliving.keys != None:  # we had a response
                    semantic_mapping_run.addData('living_nonliving.rt', living_nonliving.rt)
                    semantic_mapping_run.addData('living_nonliving.duration', living_nonliving.duration)
                # check responses
                if end_mapping.keys in ['', [], None]:  # No response was made
                    end_mapping.keys = None
                semantic_mapping_run.addData('end_mapping.keys',end_mapping.keys)
                if end_mapping.keys != None:  # we had a response
                    semantic_mapping_run.addData('end_mapping.rt', end_mapping.rt)
                    semantic_mapping_run.addData('end_mapping.duration', end_mapping.duration)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if item.maxDurationReached:
                    routineTimer.addTime(-item.maxDuration)
                elif item.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.500000)
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'semantic_mapping_run'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'runs'
        
    # completed 2.0 repeats of 'blocks'
    
    
    # --- Prepare to start Routine "thanks" ---
    # create an object to store info about Routine thanks
    thanks = data.Routine(
        name='thanks',
        components=[thanks_text],
    )
    thanks.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for thanks
    thanks.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thanks.tStart = globalClock.getTime(format='float')
    thanks.status = STARTED
    thisExp.addData('thanks.started', thanks.tStart)
    thanks.maxDuration = None
    # keep track of which components have finished
    thanksComponents = thanks.components
    for thisComponent in thanks.components:
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
    
    # --- Run Routine "thanks" ---
    thanks.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanks_text* updates
        
        # if thanks_text is starting this frame...
        if thanks_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanks_text.frameNStart = frameN  # exact frame index
            thanks_text.tStart = t  # local t and not account for scr refresh
            thanks_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanks_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thanks_text.started')
            # update status
            thanks_text.status = STARTED
            thanks_text.setAutoDraw(True)
        
        # if thanks_text is active this frame...
        if thanks_text.status == STARTED:
            # update params
            pass
        
        # if thanks_text is stopping this frame...
        if thanks_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thanks_text.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                thanks_text.tStop = t  # not accounting for scr refresh
                thanks_text.tStopRefresh = tThisFlipGlobal  # on global time
                thanks_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thanks_text.stopped')
                # update status
                thanks_text.status = FINISHED
                thanks_text.setAutoDraw(False)
        
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
            thanks.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanks.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks" ---
    for thisComponent in thanks.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thanks
    thanks.tStop = globalClock.getTime(format='float')
    thanks.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thanks.stopped', thanks.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thanks.maxDurationReached:
        routineTimer.addTime(-thanks.maxDuration)
    elif thanks.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
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
