#include <utility>
#include "state.hpp"
#include "alpha_beta_pruning.hpp"


/*============================================================
 * AlphaBetaPruning -- eval_ctx
 *
 * Negamax with alpha-beta pruning. Caller manages memory.
 * alpha: best score the maximizing side can already guarantee.
 * beta:  best score the minimizing side can already guarantee.
 *============================================================*/
int AlphaBetaPruning::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const ABParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        );
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop with Alpha-Beta pruning === */
    int best_score = M_MAX;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int raw;
        if(same){
            raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, alpha, beta);
        }else{
            raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
        }

        int score = same ? raw : -raw;
        delete next;

        if(score > best_score){
            best_score = score;
        }

        if(score > alpha){
            alpha = score;
        }

        if(alpha >= beta){
            // 觸發 Alpha-Beta 剪枝：無須繼續探索此分支。
            break;
        }
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * AlphaBetaPruning -- search
 *
 * Iterate legal moves, call eval_ctx with initial alpha/beta,
 * and return SearchResult.
 *============================================================*/
SearchResult AlphaBetaPruning::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    ABParams p = ABParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    int alpha = M_MAX;
    int beta = P_MAX;
    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int raw;
        if(same){
            raw = eval_ctx(next, depth - 1, history, 1, ctx, p, alpha, beta);
        }else{
            raw = eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha);
        }

        int score = same ? raw : -raw;
        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }

        if(score > alpha){
            alpha = score;
        }

        if(alpha >= beta){
            // 觸發 Alpha-Beta 剪枝：根節點已達 beta 界線，停止檢查剩餘走法。
            break;
        }

        move_index++;
    }

    result.score = best_score;
    result.seldepth = ctx.seldepth;
    result.nodes = ctx.nodes;
    if(total_moves > 0){
        result.pv = {result.best_move};
    }

    return result;
}


/*============================================================
 * AlphaBetaPruning -- default_params / param_defs
 *============================================================*/
ParamMap AlphaBetaPruning::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> AlphaBetaPruning::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
